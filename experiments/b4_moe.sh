#!/bin/bash
# B4: does gang-MoE beat the non-gang MoE path?  The plan's most basic Part B
# question, and one the README never answers.
set -u
cd /root/projects/fleet-chiplet-megakernel || exit 1

export AMDGPU_TARGETS=gfx942
export MIRAGE_HOME=/root/projects/fleet-chiplet-megakernel
export MIRAGE_ROOT=$MIRAGE_HOME
export HIP_VISIBLE_DEVICES=0

PROMPT="Explain the key differences between RISC and CISC processor architectures, including their instruction set design philosophies, pipeline implementations, memory access patterns, register file organization, microcode usage, and how modern processors have evolved to blur the traditional boundaries between these two approaches in terms of performance, power efficiency, and overall transistor utilization."

run() {
  local tag="$1" gang="$2" gangmoe="$3" bs="$4"
  echo "=== $tag  (USE_GANG=$gang USE_GANG_MOE=$gangmoe bs=$bs) ==="
  rm -rf permanent_output_dir
  USE_GANG="$gang" USE_GANG_MOE="$gangmoe" \
    /root/venvs/fleet/bin/python demo/qwen3/demo_30B_A3B.py --use-mirage \
    --model Qwen/Qwen3-30B-A3B \
    --max-num-batched-tokens "$bs" --max-num-batched-requests "$bs" \
    --max-seq-length 1400 --max-new-tokens 256 --ignore-eos \
    --prompt "$PROMPT" > "/tmp/moe_$tag.log" 2>&1
  echo "  exit=$? $(grep -oE 'Decode: [0-9]+ tokens.*' /tmp/moe_$tag.log | tail -1)"
}

# bs=32 is the batch we know MPerBlock=16 tiles can legalise.
run bs32_gangmoe   1 1 32
run bs32_gangonly  1 0 32
run bs32_nogang    0 0 32
