#!/usr/bin/env bash
# Run egg-toolbox with tinygrad + Qwen3-8B-Q4_0.
#
# Defaults are tuned for development/debugging on this workstation:
# modest tinygrad verbosity, BEAM=2 for decode throughput after warmup,
# and all three egg-toolbox debug surfaces on so you can see what's
# happening per-request without drowning in per-kernel-launch noise.
# Override any of these on the CLI, e.g. ``DEBUG=0 JITBEAM=0 ./run_test_server.sh``.
#
# -- tinygrad --
#   DEBUG=1          prints each kernel compile + CACHE MISS/HIT <hash>
#                    lines.  DEBUG=0 = silent, DEBUG=2 = per-launch
#                    timing spam.  With EGG_LOG_FORWARD below, DEBUG=1
#                    is the sweet spot -- you see which kernels were
#                    just compiled vs reused, adjacent to the egg
#                    forward that triggered them.
#   JITBEAM=2        beam kernel search depth.  With --keep-packed and
#                    a working cache.db (delete cache.db-wal if a prior
#                    run was SIGKILL'd mid-transaction, see
#                    workspace-2gi), BEAM=2 produces ~40 tok/s decode
#                    on Qwen3-8B-Q4_0 on a 4090.  BEAM=0 falls back to
#                    untuned kernels (~0.1-0.2 tok/s on the same
#                    workload -- use only for cache-corruption recovery).
#   BEAM_DEBUG=0     (only relevant with JITBEAM>0; set =1 to see
#                    the beam search's per-step kernel candidates)
#
# -- egg-toolbox debug surfaces --
#   EGG_LOG_FORWARD=1       per-request forward call log + summary.
#                           One line per model() call with purpose,
#                           T, start_pos, elapsed; summary line at
#                           end reports aggregate counts/timings.
#   EGG_DEBUG_PREFIX=1      prefix-cache diagnostics.  Prints the
#                           common_prefix length and, on mismatch, a
#                           window of tokens around the divergence.
#   EGG_DEBUG_MESSAGES=1    input/render diagnostics.  Prints each
#                           incoming ChatMessage (role, content_len,
#                           reasoning presence) plus the last 400 chars
#                           of the rendered prompt.
#   EGG_PREFILL_CHUNK=128   chunked-prefill chunk size.  Default 128.
#                           =0 disables chunking (single-shot prefill,
#                           slower on novel T values because kernels
#                           recompile).
#
# -- CPU quota --
# Wrapped in systemd-run --scope -p CPUQuota=400% because the bare
# tinygrad+python process hammers every core and this machine gets
# thermally/power-unstable under unbounded CPU load.  The 400% cap is
# ~4 cores worth -- tinygrad's beam search + kernel compile pool is
# CPU-bound on spawn workers, and leaving it uncapped trips the
# board's throttling or crashes the desktop session.
#
# First request compiles many kernels -- each line like
#   "*** CUDA 1234  E_... arg N mem X GB  tm 12.3us/45.6ms"
# tells you a kernel executed.  "CACHE MISS <hash>" means a new
# compile; "CACHE HIT <hash>" means reusing a cached kernel.
# Watch the log grow -- if lines stop appearing for >30s, something
# is stuck.
set -euo pipefail

# tinygrad
export DEBUG="${DEBUG:-1}"
export JITBEAM="${JITBEAM:-2}"
export BEAM_DEBUG="${BEAM_DEBUG:-0}"

# egg-toolbox debug surfaces
export EGG_LOG_FORWARD="${EGG_LOG_FORWARD:-1}"
export EGG_DEBUG_PREFIX="${EGG_DEBUG_PREFIX:-1}"
export EGG_DEBUG_MESSAGES="${EGG_DEBUG_MESSAGES:-1}"
export EGG_PREFILL_CHUNK="${EGG_PREFILL_CHUNK:-128}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec systemd-run --scope --user -p CPUQuota=400% \
  "$SCRIPT_DIR/.venv/bin/python" -m egg_toolbox "$SCRIPT_DIR/models/Qwen_Qwen3-8B-Q4_0.gguf" \
  --backend tinygrad \
  --host 127.0.0.1 --port 8765 \
  --context-length 8196 \
  --keep-packed
#  --disable-thinking   # user opt-in; do NOT enable by default
