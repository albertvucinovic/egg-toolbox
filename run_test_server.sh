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
#   EGG_JIT_CHUNKS=1        EXPERIMENTAL: make T>1 prefill chunks
#                           JIT-eligible via a symbolic-friendly
#                           causal mask (arange+where replaces triu,
#                           which rejects symbolic shape dims).  When
#                           enabled: ONE symbolic JIT trace covers
#                           every chunk position, ~680 per-kernel
#                           Python dispatches per chunk collapse into
#                           one submission, and the unique-kernel
#                           count drops from ~300 to ~100 (big impact
#                           on first-process module-load cost too).
#                           Opt-in because it patches tinygrad
#                           internals; see workspace-0j8 brittleness
#                           report.  Fallback on any failure: disable
#                           this flag.  Expected chunk time: ~400ms
#                           (GPU-bound floor) instead of ~8s.
#   EGG_SCHEDULE_CACHE=1    EXPERIMENTAL: persist tinygrad's actual
#                           schedule_cache dict to disk and reload on
#                           startup.  Near-instant scheduling on
#                           restart rather than the per-position
#                           replay cost of EGG_WARMUP=persisted.
#                           Opt-in because it depends on tinygrad-
#                           internals stability -- a version bump
#                           can break the pickle format (discarded
#                           silently in that case, falls back to
#                           position replay).
#   EGG_SCHEDULE_CACHE_FILE path to the pickle file.  Default:
#                           $XDG_CACHE_HOME/egg-toolbox/
#                           schedule-cache.pkl.
#   EGG_WARMUP=persisted    warm tinygrad's in-memory schedule_cache
#                           at load time so the first user request
#                           doesn't pay ~12s per chunk position.
#                           Modes:
#                           - ``off``       skip warmup entirely.
#                           - ``first``     chunk @ 0 + T=1 decode
#                                           (~12s added to load).
#                           - ``full``      every chunk-aligned
#                                           position up to max_context
#                                           (~minutes added to load).
#                           - ``persisted`` replay the chunk positions
#                                           this workload actually used
#                                           in prior runs (read from
#                                           $XDG_CACHE_HOME/egg-toolbox/
#                                           warmup-positions.json).
#                                           Default.  Falls back to
#                                           ``first`` on missing/
#                                           mismatched file.  Bounds
#                                           steady-state per-restart
#                                           cost to what this user's
#                                           conversations actually hit.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tinygrad's default kernel cache path is ~/.cache/tinygrad/cache.db
# (via XDG_CACHE_HOME).  We let it use that default unless the user
# already has a bigger cache somewhere we recognise -- in which case
# we pick the bigger one so prior compile work isn't thrown away.
# Candidates considered in order of preference:
#   1. ``$CACHEDB``              -- user's explicit override, wins.
#   2. ``~/.cache/tinygrad/cache.db``  -- tinygrad default, normal place.
#   3. ``$SCRIPT_DIR/.tinygrad-cache/cache.db``  -- repo-local, from
#      a brief experiment that pinned cache here.  Kept as a fallback
#      source.
# The candidate with the largest non-zero size wins; if all are empty
# or missing, we use the tinygrad default.
_size_of() { [[ -f "$1" ]] && stat -c '%s' "$1" || echo 0; }

_default_cache="${XDG_CACHE_HOME:-$HOME/.cache}/tinygrad/cache.db"
_repo_cache="$SCRIPT_DIR/.tinygrad-cache/cache.db"

if [[ -n "${CACHEDB:-}" ]]; then
  : # user explicit override, keep
else
  _default_size=$(_size_of "$_default_cache")
  _repo_size=$(_size_of "$_repo_cache")
  if (( _repo_size > _default_size )); then
    export CACHEDB="$_repo_cache"
  else
    export CACHEDB="$_default_cache"
  fi
fi
mkdir -p "$(dirname "$CACHEDB")"

# CACHELEVEL=2 is the default but be explicit.  =0 disables the disk
# cache entirely (don't use unless recovering from a corrupt cache.db).
export CACHELEVEL="${CACHELEVEL:-2}"

# Report cache state at startup so we can see definitively whether
# the cache file exists and has data, before any kernel tries to
# read from it.  Shows BOTH the default and the repo-local paths so
# you can see if there's a cache hiding somewhere we're not using.
echo "=== tinygrad kernel cache state ==="
printf '  active path : %s\n' "$CACHEDB"
for candidate in "$_default_cache" "$_repo_cache"; do
  if [[ -f "$candidate" ]]; then
    size_bytes=$(stat -c '%s' "$candidate")
    mtime=$(stat -c '%y' "$candidate")
    marker=""
    [[ "$candidate" == "$CACHEDB" ]] && marker=" <-- using this one"
    printf '  %-50s  %12s bytes  %s%s\n' \
      "$candidate" "$size_bytes" "${mtime%%.*}" "$marker"
    for suffix in -wal -shm; do
      if [[ -f "${candidate}${suffix}" ]]; then
        s=$(stat -c '%s' "${candidate}${suffix}")
        printf '  %-50s  %12s bytes\n' "${candidate}${suffix}" "$s"
      fi
    done
  fi
done
if [[ ! -f "$CACHEDB" ]]; then
  echo "  (active cache does not exist yet -- first run, expect full BEAM compile)"
fi
echo "==================================="

# tinygrad verbosity + beam
export DEBUG="${DEBUG:-1}"
export JITBEAM="${JITBEAM:-2}"
export BEAM_DEBUG="${BEAM_DEBUG:-0}"

# egg-toolbox debug surfaces
export EGG_LOG_FORWARD="${EGG_LOG_FORWARD:-1}"
export EGG_DEBUG_PREFIX="${EGG_DEBUG_PREFIX:-1}"
export EGG_DEBUG_MESSAGES="${EGG_DEBUG_MESSAGES:-1}"
export EGG_PREFILL_CHUNK="${EGG_PREFILL_CHUNK:-128}"
export EGG_WARMUP="${EGG_WARMUP:-persisted}"
# Experimental: full schedule_cache persistence.  Opt-in.  If it
# works on your workload, near-instant restart.  If tinygrad changes
# its schedule_cache internals in a way that breaks the pickle
# format, the load step logs and discards silently and we fall back
# to position-replay.
export EGG_SCHEDULE_CACHE="${EGG_SCHEDULE_CACHE:-1}"
# Experimental: JIT-eligible T>1 prefill chunks via symbolic-mask
# attention.  Patches each TransformerBlock's _attention method to
# use arange+where instead of triu.  Default 1 to try it; revert to
# 0 if something's broken at startup or first chunk forward.
export EGG_JIT_CHUNKS="${EGG_JIT_CHUNKS:-1}"

# Force Python's stdout/stderr unbuffered so tinygrad's DEBUG lines
# and our EGG_* prints flush immediately rather than accumulating in
# a pipe buffer.  Without this, running under systemd-run --scope
# (which may not attach the child to a TTY) makes prints invisible
# until a buffer fills -- looks like "nothing's happening" for minutes.
export PYTHONUNBUFFERED=1

# Explicitly forward env vars through systemd-run using the
# ``--setenv=VAR=$VAR`` form rather than the bare ``--setenv=VAR``
# form.  The bare form is supposed to propagate the current value but
# has been observed to be inconsistent across systemd versions / under
# --user scopes.  Explicit ``=VAR=value`` is unambiguous.
exec systemd-run --scope --user -p CPUQuota=400% \
  --setenv="CACHEDB=$CACHEDB" \
  --setenv="CACHELEVEL=$CACHELEVEL" \
  --setenv="HOME=$HOME" \
  --setenv="XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}" \
  --setenv="DEBUG=$DEBUG" \
  --setenv="JITBEAM=$JITBEAM" \
  --setenv="BEAM_DEBUG=$BEAM_DEBUG" \
  --setenv="EGG_LOG_FORWARD=$EGG_LOG_FORWARD" \
  --setenv="EGG_DEBUG_PREFIX=$EGG_DEBUG_PREFIX" \
  --setenv="EGG_DEBUG_MESSAGES=$EGG_DEBUG_MESSAGES" \
  --setenv="EGG_PREFILL_CHUNK=$EGG_PREFILL_CHUNK" \
  --setenv="EGG_WARMUP=$EGG_WARMUP" \
  --setenv="EGG_SCHEDULE_CACHE=$EGG_SCHEDULE_CACHE" \
  --setenv="EGG_JIT_CHUNKS=$EGG_JIT_CHUNKS" \
  --setenv="PYTHONUNBUFFERED=$PYTHONUNBUFFERED" \
  "$SCRIPT_DIR/.venv/bin/python" -u -m egg_toolbox "$SCRIPT_DIR/models/Qwen_Qwen3-8B-Q4_0.gguf" \
  --backend tinygrad \
  --host 127.0.0.1 --port 8765 \
  --context-length 8196 \
  --keep-packed
#  --disable-thinking   # user opt-in; do NOT enable by default
