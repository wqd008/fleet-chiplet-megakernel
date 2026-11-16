#!/bin/bash
# Smoke test per plan section 1.6 -- the single gate before any search.
set -u
cd /root/projects/fleet-chiplet-megakernel || exit 1

export AMDGPU_TARGETS=gfx942
export MIRAGE_HOME=/root/projects/fleet-chiplet-megakernel
export MIRAGE_ROOT=/root/projects/fleet-chiplet-megakernel
export PATH="/root/venvs/fleet/bin:$PATH"

rm -rf permanent_output_dir

PROMPT="Explain the key differences between RISC and CISC processor architectures, including their instruction set design philosophies, pipeline implementations, memory access patterns, register file organization, microcode usage, and how modern processors have evolved to blur the traditional boundaries between these two approaches in terms of performance, power efficiency, and overall transistor utilization."

HIP_VISIBLE_DEVICES=0 USE_GANG=1 USE_CK_FMHA=1 USE_FUSED_SILU=1 GANG_M_TILES_GATEUP=1 \
  /root/venvs/fleet/bin/python demo/qwen3/demo.py --use-mirage \
  --model Qwen/Qwen3-8B \
  --max-num-batched-tokens 1 --max-num-batched-requests 1 \
  --max-seq-length 1200 --max-new-tokens 1024 --ignore-eos \
  --prompt "$PROMPT"
