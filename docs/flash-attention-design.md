# Block-Tiled FlashAttention for egg-toolbox

Status: design, not yet implemented.
Owner: albert.vucinovic@gmail.com
Epic: workspace-TBD (filed alongside this doc)

## Motivation

tinygrad's BEAM kernel-search chokes on kernels with a symbolic axis: it
enumerates `UPCAST`/`UNROLL` actions per axis, but for an axis of size
`start_pos+T` (bound at JIT-run time, unknown at compile time) it can't
tell which actions are valid and must time each candidate at a midpoint
value that's close to the worst case. On Qwen3-8B the symbolic
attention kernel takes many minutes of BEAM search per variant and in
practice triggers either timeouts, deadlocks on sqlite WAL, or (on
13th-gen Intel) hardware-faults under sustained compile load.

The upstream `_attention` path today:

```python
mask = Tensor.full((1, 1, T, start_pos+T), -inf).triu(start_pos+1)
attn = q @ k.transpose(-2,-1) + mask      # (B, H, T, start_pos+T)   <- symbolic
attn = attn.softmax(-1) @ v               # (B, H, T, head_dim)
```

Everything that touches the symbolic `start_pos+T` axis defeats BEAM.

llama.cpp's CUDA path side-steps this by structuring attention as a
loop over fixed-size K/V blocks (FlashAttention-1 tiling). Each inner
block is a fully-int-shape kernel a compiler can optimize perfectly.
We can do the same in egg-toolbox without upstream tinygrad changes.

## Approach

Restructure `_attention` as:

1. Compute `Q, K, V` for the current chunk (unchanged).
2. Write new `K, V` into `cache_kv[:, :, :, start_pos:start_pos+T, :]`
   (unchanged, symbolic slice assignment already works).
3. **New**: iterate over K/V cache in fixed-size blocks of
   `B_block` (default 256) positions. For each block:
   - Load `cache_kv[:, :, :, block_start:block_end, :]` — fixed shape.
   - Run one JIT-captured, BEAM-tuned "block attention" kernel that
     updates the running online-softmax state `(m, l, out)`.
4. After the loop, divide `out` by `l` to finalize.

The Python-side loop count is a concrete int at call time
(`ceil((start_pos+T)/B_block)`). The inner kernel has only int dims,
so BEAM handles it. The same JIT-captured inner kernel serves every
block, every layer, every forward pass.

## Online-softmax math (FlashAttention-1)

Carry `(m_running, l_running, out_running)` across blocks. For each
block compute:

```
s       = q @ k_block.transpose(-2,-1) / sqrt(d)   # (B, H, T, B_block)
s       = s + mask                                  # -inf for masked positions (optional)
m_block = max(s, dim=-1, keepdim=True)              # (B, H, T, 1)
m_new   = maximum(m_running, m_block)               # (B, H, T, 1)
alpha   = exp(m_running - m_new)                    # correction for accumulator
p_tilde = exp(s - m_new)                            # (B, H, T, B_block), un-normalized
l_new   = l_running * alpha + p_tilde.sum(-1, keepdim=True)
out_new = out_running * alpha + p_tilde @ v_block
```

At loop end:

```
out_final = out_running / l_running
```

Initial state: `m_running = -inf`, `l_running = 0`, `out_running = 0`.

Numerical stability: both `exp(m_running - m_new)` and `exp(s - m_new)`
are ≤ 1 by construction (subtracting the running maximum before
exponentiating). Runs in fp32 internally, cast to model dtype at
finalize.

## Block classification

Given the current chunk has queries at positions
`[start_pos, start_pos+T)` and attends to keys at `[0, start_pos+T)`:

- **FULL block**: `block_end <= start_pos`. All queries in the chunk
  attend to all keys in this block. No mask needed.
- **BOUNDARY block**: `block_start < start_pos+T < block_end` OR the
  block that contains `start_pos..start_pos+T-1`. Causal masking
  applies: for query at chunk-position `i`, key at block-local-position
  `j` is attended iff `block_start + j <= start_pos + i`.
- **RAGGED block**: final block where `block_end > start_pos+T`.
  Positions `>= start_pos+T` are invalid (cache_kv slot is
  zero-initialized or stale). Must be masked to prevent spurious
  contribution to softmax normalizer.
- **FUTURE block**: `block_start >= start_pos+T`. Entirely invalid.
  Skipped Python-side — never dispatched.

For our chunked-prefill path, classification reduces to:
- Blocks 0..(floor((start_pos-1)/B_block)): FULL.
- At most one BOUNDARY block covering the chunk's own query range.
- Possibly ragged on the final block depending on alignment.

## JIT variants

Two TinyJit instances:

1. `_block_attn_full(q, k_block, v_block, m, l, out)` — no mask tensor.
2. `_block_attn_masked(q, k_block, v_block, m, l, out, mask)` — mask is
   a fixed-shape `(T, B_block)` tensor; the dispatcher constructs it
   in Python as an int-shape arange-based boolean.

Both have all-int shapes, so each is BEAM-tuneable end-to-end.

Kernel count after warmup: 2 per model × 1 shape signature per
`(T, B_block, head_dim, n_heads, n_kv_heads)`. Stable across all
positions, all chunks, all layers.

## GQA handling

Qwen3-8B has `n_heads=32, n_kv_heads=8` → group size 4. Two approaches:

- **Repeat on load**: `k, v = k.repeat_interleave(4, dim=1)` once per
  forward. Expands to `n_heads` K/V → wasteful (4× memory read), but
  simplest kernel. Done this way by tinygrad upstream via
  `enable_gqa=True`.
- **Broadcast in kernel**: reshape q as `(B, n_kv_heads, group, T, d)`
  and k_block as `(B, n_kv_heads, 1, B_block, d)` so matmul broadcasts
  naturally. Saves the K/V memory traffic inside the block kernel.

Start with approach 1 (repeat) for simplicity in milestones M1-M3;
move to approach 2 (broadcast) in M5 once everything else works.

## Block size selection

`B_block` default: **256**. Knobs to consider:

- Too small → more Python dispatch, loop overhead dominates.
- Too big → cold-compile blowup, VRAM pressure on accumulator
  buffers.
- Should divide head_dim cleanly for matmul tiling.
- Common values in production FA kernels: 64, 128, 256.

Env var: `EGG_FLASH_BLOCK_SIZE` override. Benchmark on Qwen3-8B to
pick the sweet spot.

## Causal mask construction (per-block)

For the BOUNDARY block at `block_start`, the mask is:

```python
# rows: query positions within chunk, cols: key positions within block
rows = arange(T).reshape(T, 1)                   # (T, 1)
cols = arange(B_block).reshape(1, B_block)       # (1, B_block)
# key_abs_pos = block_start + cols
# query_abs_pos = start_pos + rows
# attend iff key_abs_pos <= query_abs_pos
# <=> (block_start + cols) <= (start_pos + rows)
# <=> cols <= (start_pos + rows - block_start)
allowed = cols <= (start_pos + rows - block_start)   # (T, B_block), bool
mask = where(allowed, 0.0, -inf)
```

All dims are int at Python evaluation time (`block_start`, `start_pos`,
`T`, `B_block` are all Python ints in the dispatching loop), so this
mask has a static shape and BEAM-tunable structure.

For RAGGED blocks (last block, partial valid): same construction but
also clip `cols >= valid_length_in_block` to -inf.

## Integration points

Opt-in via env var `EGG_FLASH_ATTENTION=1` initially. When set,
`LlamaArchitecture.__init__` monkey-patches each block's `_attention`
with the flash variant (same mechanism as the current
`patch_block_attention` in `symbolic_attention.py`, which becomes
obsolete).

Once stable and perf-verified, flip the default to on. Keep an
`EGG_FLASH_ATTENTION=0` escape hatch for debugging.

## Milestones

1. **M1** — numpy reference: implement tiled online-softmax in numpy,
   verify bit-equivalence against numpy-naive `softmax(q @ k^T + mask)
   @ v` over a range of shapes and mask configurations.
2. **M2** — single-block tinygrad port: tile-attention with exactly
   one block, validate against upstream `scaled_dot_product_attention`
   at fp32.
3. **M3** — multi-block non-causal: loop over blocks, online softmax
   accumulates correctly, matches single-block reference.
4. **M4** — causal masking: boundary + ragged block handling, matches
   upstream triu-based attention at fp32 and fp16.
5. **M5** — GQA: broadcast variant replacing per-forward
   `repeat_interleave`.
6. **M6** — JIT wrapping: two TinyJit instances (full + masked),
   replay works across blocks, layers, forwards.
7. **M7** — LlamaArchitecture integration: env-var opt-in, delete
   `symbolic_attention.py`, update docs.
8. **M8** — perf measurement: chunked prefill tps vs current baseline
   (triu + JITBEAM=2 + int start_pos). Target: match or beat 40 tps;
   stretch: 100+ tps with warm BEAM.

Quality gates: each milestone ships with its own pytest coverage.
M1-M5 use fixed seeds and numerical tolerances; M6-M7 use real
tinygrad state; M8 uses a fixed-workload benchmark script.

## Risks

- **Online softmax numerical drift**: silent — regressions show up as
  wrong outputs but no exception. Mitigation: strict bit-exactness
  tests at fp32 against numpy reference, and % diff tests against
  upstream SDPA at fp16.
- **BEAM on the inner block kernel is still expensive** (but bounded
  and one-time). Cold compile could be minutes; cache.db persistence
  carries it across restarts.
- **Python dispatch per block**: at 28 layers × ceil(N/256) blocks × T
  tokens × typical latencies, dispatch overhead could be 10-50ms per
  chunk. Acceptable for prefill; marginal for T=1 decode (1 block).
  Mitigate by keeping the loop in a pure-Python tight path.
- **Block size tuning is workload-dependent**. Chat with ~4K context
  and Qwen3-8B on 4090 is the target; other configs might need a
  different B_block.

## Non-goals (for this epic)

- Paged KV cache. Block structure makes it easy later
  (workspace-bue Tier 3), but landing that now would bloat scope.
- Multi-tenant prefix caching (separate scope; see
  prefix-kv-caching bd memory).
- Backward pass (inference-only project).
- Custom CUDA kernels. We stay pure tinygrad — the whole point is
  that the inner kernel *is* a normal tinygrad graph that BEAM tunes.
