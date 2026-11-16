#!/usr/bin/env python3
"""Quantify how far Fleet's MoE decode is from the memory-bandwidth roofline.

Two separate questions, deliberately not conflated:

  (1) WEIGHT TRAFFIC -- is the kernel reading more expert weights than it must?
      At decode the MoE is strongly weight-bandwidth-bound, so this is the binding
      constraint.  The minimum is `num_activated_experts x W_expert`: every expert
      that receives at least one token must have its weights read once.  This is a
      property of the ROUTING, not of the kernel.  Fleet's per-expert dispatch
      achieves it (it reads each activated expert exactly once).

  (2) ROW UTILISATION -- what fraction of the GEMM rows it computes carry data?
      gang_moe_linear_mi300.cuh issues a dense M=BATCH_SIZE GEMM for every
      activated expert and only skips rows in the epilogue, so
          processed_rows = num_activated x BATCH_SIZE
          useful_rows    = BATCH_SIZE x topk
      This wastes compute and activation traffic, but NOT weight traffic.  Whether
      it is *payable* depends on where the kernel sits relative to the ridge point
      of the roofline, which is what this script computes.
"""
import argparse
import json
import statistics

# --- Qwen3-30B-A3B ---
HIDDEN = 2048
MOE_INTER = 768
LAYERS = 48
N_EXPERTS = 128
TOPK = 8
BYTES = 2                    # bf16
PEAK_BF16_TFLOPS = 1307.4    # MI300X bf16 dense peak
HBM_BW_BYTES = 3.7e12        # ~achievable on MI300X (plan's figure)


def experts_touched(batch, n_experts=N_EXPERTS, topk=TOPK):
    """Expected distinct experts hit when batch*topk assignments land uniformly."""
    n = batch * topk
    return n_experts * (1.0 - (1.0 - 1.0 / n_experts) ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured-ms", type=float, required=True,
                    help="measured decode ms/token for the whole model")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-activated", type=float, default=None,
                    help="override; otherwise the analytic expectation is used")
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--dense-share-ms", type=float, default=None,
                    help="measured cost of the NON-MoE part, if known")
    args = ap.parse_args()

    w_expert = (2 * MOE_INTER * HIDDEN + MOE_INTER * HIDDEN) * BYTES
    print(f"per-expert weight: W13={(2*MOE_INTER*HIDDEN)*BYTES/1e6:.2f} MB  "
          f"Wdown={(MOE_INTER*HIDDEN)*BYTES/1e6:.2f} MB  "
          f"total={w_expert/1e6:.2f} MB")
    print(f"all experts x all layers: "
          f"{args.layers*N_EXPERTS*w_expert/1e9:.1f} GB")

    na = args.num_activated if args.num_activated else experts_touched(args.batch)
    per_iter_bytes = args.layers * na * w_expert
    ideal_ms = per_iter_bytes / HBM_BW_BYTES * 1e3
    print(f"\nbatch={args.batch} topk={TOPK}")
    print(f"activated experts (analytic E[distinct]) = {na:.1f} of {N_EXPERTS} "
          f"({100*na/N_EXPERTS:.0f}%)")
    print(f"=> MoE weight traffic per iteration = {per_iter_bytes/1e9:.1f} GB "
          f"({100*per_iter_bytes/(args.layers*N_EXPERTS*w_expert):.0f}% of the "
          f"entire expert weight set)")
    print(f"=> bandwidth-bound floor = {ideal_ms:.2f} ms / iteration "
          f"({ideal_ms/args.batch:.2f} ms / token)")

    # compute side
    useful_rows = args.batch * TOPK
    processed_rows = na * args.batch
    util = 100.0 * useful_rows / processed_rows
    macs = processed_rows * (2 * MOE_INTER + MOE_INTER) * HIDDEN * args.layers
    flops = 2 * macs
    ideal_compute_ms = flops / (PEAK_BF16_TFLOPS * 1e12) * 1e3
    print(f"\nrow utilisation = {useful_rows:.0f}/{processed_rows:.0f} = {util:.1f}%"
          f"   (waste factor {processed_rows/useful_rows:.1f}x)")
    print(f"MoE GEMM work = {flops/1e12:.2f} TFLOP/iteration -> "
          f"{ideal_compute_ms:.2f} ms at peak")
    ridge = PEAK_BF16_TFLOPS * 1e12 / HBM_BW_BYTES
    print(f"arithmetic intensity of the MoE = "
          f"{flops/per_iter_bytes:.1f} FLOP/byte vs machine ridge {ridge:.0f} "
          f"FLOP/byte")
    print(f"=> the MoE is {'MEMORY-bound' if flops/per_iter_bytes < ridge else 'COMPUTE-bound'}; "
          f"wasted rows cost compute, which is "
          f"{'largely hidden' if ideal_compute_ms < ideal_ms else 'NOT hidden'}")

    # compare against measurement
    print(f"\n--- measured: {args.measured_ms:.3f} ms/token "
          f"({args.measured_ms*args.batch:.2f} ms/iteration) ---")
    it = args.measured_ms * args.batch
    print(f"MoE floor / measured iteration = {ideal_ms:.2f}/{it:.2f} = "
          f"{100*ideal_ms/it:.0f}% of the iteration is unavoidable MoE weight traffic")
    if args.dense_share_ms:
        print(f"non-MoE share (measured) = {args.dense_share_ms:.2f} ms/iteration; "
              f"MoE share = {it-args.dense_share_ms:.2f} ms")
        achieved = per_iter_bytes / ((it - args.dense_share_ms) / 1e3)
        print(f"=> achieved MoE bandwidth = {achieved/1e12:.2f} TB/s "
              f"vs {HBM_BW_BYTES/1e12:.1f} TB/s assumed")


if __name__ == "__main__":
    main()
