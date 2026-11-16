#!/bin/bash
# Bootstrap a fresh AMD ROCm image so that this repo builds and runs on MI300X.
# Companion to docs/MI300X-移植指南.md -- see that file for the *why*.
#
# Tested on: ROCm 7.14.1, Ubuntu 24.04, MI300X VF (gfx942), 304 CU.
#
# Idempotent: safe to re-run; steps that are already done are skipped.
# NOTE: no `set -e` -- several steps legitimately fail-and-continue on this
# image (see the apt/dpkg note in step 1).
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/root/venvs/fleet
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- 1. python
log "1/7 pip + venv"
if [ ! -x "$PY" ]; then
  apt-get install -y python3-pip python3.12-venv
  python3 -m venv "$VENV"
fi
# The apt run above ends with a dpkg error about linux-image-6.8.0-149-generic
# (a dkms postinst failure).  It is pre-existing and unrelated -- ignore it.
"$PY" -m pip --version | head -1

log "1b/7 torch (ROCm wheel)"
"$PY" -c "import torch" 2>/dev/null || \
  "$PIP" install --no-cache-dir "torch==2.10.0+rocm7.0" \
    --index-url https://download.pytorch.org/whl/rocm7.0
"$PY" -c "import torch;print('torch',torch.__version__,'hip',torch.version.hip)"

log "1c/7 build + python deps"
# cmake MUST stay < 4.x: deps/json declares cmake_minimum_required(3.1...3.14)
"$PIP" install -q "cmake==3.27.9" hatchling cython "z3-solver==4.15" numpy \
  graphviz tqdm protobuf accelerate==1.8.0 transformers==4.57.1 huggingface_hub
"$VENV/bin/cmake" --version | head -1

log "1d/7 rust (setup.py builds two cargo crates)"
command -v cargo >/dev/null 2>&1 || \
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="/root/.cargo/bin:$PATH"
cargo --version

# ------------------------------------------------------ 2. ROCm cmake files
log "2/7 ROCm HIP cmake package"
if [ -f /opt/rocm/lib/cmake/hip-lang/hip-lang-config.cmake ]; then
  echo "already present"
else
  # apt/dpkg is broken on this image, so bypass it entirely with dpkg-deb -x.
  mkdir -p /root/rocm-debs && cd /root/rocm-debs
  # match the version to the /opt/rocm/core-X.Y directory that exists
  ROCMVER="$(ls -d /opt/rocm/core-* 2>/dev/null | head -1 | sed 's#.*/core-##')"
  echo "detected ROCm $ROCMVER"
  apt-get download "amdrocm-runtime-dev${ROCMVER}" && \
    dpkg-deb -x "amdrocm-runtime-dev${ROCMVER}"_*_amd64.deb /
  ls /opt/rocm/lib/cmake/hip-lang/hip-lang-config.cmake
fi

# ------------------------------------------------------------- 3. git deps
log "3/7 deps: json / cutlass / composable_kernel"
cd "$REPO"
fetch_dep() {  # name url sha
  local name="$1" url="$2" sha="$3"
  if [ -d "deps/$name/.git" ]; then echo "$name already fetched"; return 0; fi
  mkdir -p "deps/$name" && cd "deps/$name" || return 1
  git init -q . && git remote add origin "$url"
  git fetch -q --depth 1 origin "$sha" && git checkout -q FETCH_HEAD
  echo "$name @ $(git rev-parse --short HEAD)"
  cd "$REPO"
}
# SHAs taken from upstream mirage-project/mirage's gitlinks (authoritative).
fetch_dep json    https://github.com/nlohmann/json.git   8c391e04fe4195d8be862c97f38cfe10e2a3472e
fetch_dep cutlass https://github.com/NVIDIA/cutlass.git   f3fde58372d33e9a5650ba7b80fc48b3b49d40c8

# composable_kernel: MUST be this exact commit.  Fleet calls
# TileFmhaFwdSplitKVTraits with 14 template args (needs kHasSink), but its
# policies do NOT implement the newer GemmPipelineAGmemBGmemCRegV2 requirements
# (MakeADramTensorView / Policy::is_a_load_tr).  ac18460782fa is the newest
# commit satisfying both; the next commit on develop breaks the build.
CK_SHA=ac18460782fadcd24aa321394c63dc284c90593b
if [ -d deps/composable_kernel/include ]; then
  echo "composable_kernel already fetched"
else
  mkdir -p deps/composable_kernel && cd deps/composable_kernel
  git init -q . && git remote add origin https://github.com/ROCm/composable_kernel.git
  git sparse-checkout init --cone && git sparse-checkout set include
  git fetch -q --depth 1 origin "$CK_SHA" && git checkout -q FETCH_HEAD
  echo "composable_kernel @ $(git rev-parse --short HEAD)"
  cd "$REPO"
fi
# deps/z3 is NOT needed -- setup.py uses the z3-solver wheel.

# ----------------------------------------------------------------- 4. build
log "4/7 build"
export AMDGPU_TARGETS=gfx942 ROCM_PATH=/opt/rocm
export PATH="/root/.cargo/bin:$VENV/bin:$PATH"
cd "$REPO" && rm -rf build
"$PIP" install -e . -v --no-build-isolation 2>&1 | tail -5
"$PY" -c "import mirage; print('mirage OK')"

# ------------------------------------------------------------- 5. verify
log "5/7 smoke test (dense path)"
export MIRAGE_HOME="$REPO" MIRAGE_ROOT="$REPO" HIP_VISIBLE_DEVICES=0
cd "$REPO" && rm -rf permanent_output_dir
PROMPT="Explain the key differences between RISC and CISC processor architectures, including their instruction set design philosophies, pipeline implementations, memory access patterns, register file organization, microcode usage, and how modern processors have evolved to blur the traditional boundaries between these two approaches in terms of performance, power efficiency, and overall transistor utilization."
USE_GANG=1 USE_CK_FMHA=0 USE_FUSED_SILU=1 GANG_M_TILES_GATEUP=1 \
  "$PY" demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-8B \
  --max-num-batched-tokens 1 --max-num-batched-requests 1 \
  --max-seq-length 1200 --max-new-tokens 1024 --ignore-eos --prompt "$PROMPT" \
  2>&1 | grep -E "Decode:|error" | tail -3

log "done"
cat <<'EOF'

Environment ready.  Export these in every shell before running:

    export AMDGPU_TARGETS=gfx942            # mandatory; in-tree default is gfx950
    export MIRAGE_HOME=<repo>
    export MIRAGE_ROOT=$MIRAGE_HOME
    export PATH=/root/venvs/fleet/bin:$PATH

And ALWAYS `rm -rf permanent_output_dir` before changing any parameter --
the megakernel is compiled per mode and a stale directory silently reuses it.
EOF
