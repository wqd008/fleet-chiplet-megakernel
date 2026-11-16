#!/bin/bash
# Part B smoke test: does the MoE path run at all on MI300X?
# demo_30B_A3B.py already defaults USE_CK_FMHA=0 (unlike demo.py), so no
# adaptation is needed for Part B.
set -u
cd /root/projects/fleet-chiplet-megakernel || exit 1

export AMDGPU_TARGETS=gfx942
export MIRAGE_HOME=/root/projects/fleet-chiplet-megakernel
export MIRAGE_ROOT=$MIRAGE_HOME
export HIP_VISIBLE_DEVICES=0

rm -rf permanent_output_dir

PROMPT="Explain the key differences between RISC and CISC processor architectures, including their instruction set design philosophies, pipeline implementations, memory access patterns, register file organization, microcode usage, and how modern processors have evolved to blur the traditional boundaries between these two approaches in terms of performance, power efficiency, and overall transistor utilization."

USE_GANG=1 USE_GANG_MOE="${USE_GANG_MOE:-1}" \
  /root/venvs/fleet/bin/python demo/qwen3/demo_30B_A3B.py --use-mirage \
  --model Qwen/Qwen3-30B-A3B \
  --max-num-batched-tokens "${BS:-1}" --max-num-batched-requests "${BS:-1}" \
  --max-seq-length 1400 --max-new-tokens "${MAXNEW:-256}" --ignore-eos \
  --prompt "$PROMPT"
