#!/usr/bin/env bash
# Run egg-toolbox with tinygrad + Qwen3-8B-Q4_0.
#
# Env defaults (override on CLI: `DEBUG=1 ./run_test_server.sh`):
#
#   DEBUG=2          tinygrad prints each kernel compile + launch.
#                    DEBUG=1 -> just compiles; DEBUG=0 -> silent.
#   JITBEAM=0        beam search disabled.
#                    !!! DO NOT use JITBEAM>0 with --keep-packed !!!
#                    It deadlocks on the sqlite cache.db under
#                    concurrent workers (observed on tinygrad
#                    ~2026-04).  See workspace-... bd issue.
#                    Decode will be slow (~0.17 tok/s on 4090 for
#                    Qwen3-8B-Q4_0) but it WILL produce tokens
#                    with progress visible in the log.
#   BEAM_DEBUG=0     (only relevant if JITBEAM>0)
#
# First request compiles many kernels -- each line like
#   "*** CUDA 1234  E_... arg N mem X GB  tm 12.3us/45.6ms"
# tells you a kernel executed.  "CACHE MISS <hash>" means a new
# compile; "CACHE HIT <hash>" means reusing a cached kernel.
# Watch the log grow -- if lines stop appearing for >30s, something
# is stuck.
set -euo pipefail

export DEBUG="${DEBUG:-2}"
export JITBEAM="${JITBEAM:-0}"
export BEAM_DEBUG="${BEAM_DEBUG:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python" -m egg_toolbox "$SCRIPT_DIR/models/Qwen_Qwen3-8B-Q4_0.gguf" \
  --backend tinygrad \
  --host 127.0.0.1 --port 8765 \
  --context-length 2048 \
  --disable-thinking
#  --keep-packed
