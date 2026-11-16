#!/bin/bash
# Validity gate: does the MI300X gang path actually produce the same tokens as
# the PyTorch reference?  A fast baseline is worthless if the outputs are wrong.
set -u
cd /root/projects/fleet-chiplet-megakernel || exit 1

export AMDGPU_TARGETS=gfx942
export MIRAGE_HOME=/root/projects/fleet-chiplet-megakernel
export MIRAGE_ROOT=$MIRAGE_HOME
export HIP_VISIBLE_DEVICES=0

PROMPT="Explain the key differences between RISC and CISC processor architectures, including their instruction set design philosophies, pipeline implementations, memory access patterns, register file organization, microcode usage, and how modern processors have evolved to blur the traditional boundaries between these two approaches in terms of performance, power efficiency, and overall transistor utilization."

COMMON=(--model Qwen/Qwen3-8B
        --max-num-batched-tokens 1 --max-num-batched-requests 1
        --max-seq-length 1200 --max-new-tokens 64 --ignore-eos
        --prompt "$PROMPT")

echo "### 1/2 PyTorch reference (no --use-mirage)"
/root/venvs/fleet/bin/python demo/qwen3/demo.py \
  "${COMMON[@]}" --save-tokens /tmp/torch_tok.json > /tmp/corr_torch.log 2>&1
echo "  exit=$?"

echo "### 2/2 Mirage MPK, gang + USE_CK_FMHA=0"
rm -rf permanent_output_dir
USE_GANG=1 USE_CK_FMHA=0 USE_FUSED_SILU=1 GANG_M_TILES_GATEUP=1 \
  /root/venvs/fleet/bin/python demo/qwen3/demo.py --use-mirage \
  "${COMMON[@]}" --save-tokens /tmp/mpk_tok.json > /tmp/corr_mpk.log 2>&1
echo "  exit=$?"
