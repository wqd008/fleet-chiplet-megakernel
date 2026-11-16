#!/usr/bin/env python3
"""B1/B2: per-expert token histogram and MoE row utilisation for Qwen3-30B-A3B.

Why host-side rather than reading the megakernel's buffers: `moe_routing_indices`
is allocated with `mpk.new_tensor(..., io_category="cuda_tensor")`, which maps to
`attach_cuda_tensor`, and there is no python-side readback for those.  Routing is a
deterministic function of the hidden state, so re-deriving it from the same weights
with hooks on the router gives the same statistics, and can be cross-checked later.

What this measures, and why it is the crux of the "is Fleet's MoE naive?" question:

  gang_moe_linear_mi300.cuh issues, for EVERY activated expert, a dense M= BATCH_SIZE
  GEMM, and only skips rows in the epilogue (`if (route_val != 0)`).  So the work
  actually performed is

      processed_rows = num_activated_experts x BATCH_SIZE

  whereas the useful work is

      useful_rows    = BATCH_SIZE x topk

  and utilisation = useful/processed = topk / num_activated_experts.

  NOTE this is a compute/activation-traffic waste, NOT a weight-traffic waste: each
  activated expert's weights must be read at least once regardless of how few tokens
  hit it.  That distinction decides whether the waste is actually payable.
"""
import argparse
import json
import statistics
from collections import Counter

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=64, help="tokens per request")
    ap.add_argument("--out", default="/root/projects/fleet-exp/results/moe_routing.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.eval()

    cfg = model.config
    n_experts = cfg.num_experts
    topk = cfg.num_experts_per_tok
    print(f"experts={n_experts} topk={topk} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} moe_inter={cfg.moe_intermediate_size}")

    # Genuinely diverse prompts.  A batch of near-identical prompts collapses the
    # routing histogram (every sequence ends on the same token), which would make
    # the decode view look far better than it is in real serving.
    topics = [
        "Write a Python function that merges two sorted lists.",
        "Explain the difference between TCP and UDP.",
        "What is the capital of France and why is it significant?",
        "Summarize the theory of relativity in simple terms.",
        "How do vaccines train the immune system?",
        "Describe the plot of Hamlet in three paragraphs.",
        "What causes earthquakes along tectonic plate boundaries?",
        "Compare functional and object-oriented programming paradigms.",
        "Explain how a transformer attention mechanism works mathematically.",
        "What are the main differences between Buddhism and Hinduism?",
        "How does a refrigerator work thermodynamically?",
        "List the key events of the French Revolution.",
        "Explain quantum entanglement to a high school student.",
        "What is the significance of the Turing test in AI?",
        "Describe the water cycle and its role in climate.",
        "How do stock options differ from restricted stock units?",
        "What are the health effects of a Mediterranean diet?",
        "Explain the CAP theorem for distributed databases.",
        "How does CRISPR gene editing work?",
        "What were the causes of the 2008 financial crisis?",
        "Describe the structure of a bacterial cell.",
        "Explain how GPS determines your position.",
        "What is the difference between a compiler and an interpreter?",
        "How do noise-cancelling headphones work?",
        "Summarize the history of the Roman Empire.",
        "What is dark matter and why do physicists think it exists?",
        "Explain the rules of cricket to someone who has never seen it.",
        "How does the stock market affect inflation?",
        "What are the benefits of renewable energy sources?",
        "Describe how neural networks learn through backpropagation.",
        "Explain the concept of opportunity cost in economics.",
        "What is the role of mitochondria in cellular respiration?",
    ]
    prompts = (topics * (args.batch // len(topics) + 1))[: args.batch]
    enc = tok(prompts, return_tensors="pt", padding="max_length",
              truncation=True, max_length=args.seq).to("cuda")
    last_idx = enc["attention_mask"].sum(dim=1).clamp(min=1).long() - 1
    print("last real token index per request:", last_idx[:8].tolist(), "...")

    # The gate is called with a flattened (B*S, hidden) view in the MoE forward,
    # or (B, S, hidden) depending on the transformers version -- handle both.
    # Two views are collected:
    #   decode -- the last position of each request, i.e. the distribution a
    #             single decode step actually sees (this is the one that matters
    #             for Fleet's MoE kernel geometry)
    #   allpos -- every position, a much larger sample of the routing distribution
    hits = {li: Counter() for li in range(cfg.num_hidden_layers)}
    allpos = {li: Counter() for li in range(cfg.num_hidden_layers)}

    def reduce_last(out):
        # last REAL token of each request -- not the last padded slot, which
        # would make every request identical and collapse the histogram
        if out.dim() == 3:
            t = out
        elif out.shape[0] % args.batch == 0:
            t = out.reshape(args.batch, -1, out.shape[-1])
        else:
            return out[-1:, :]
        ar = torch.arange(args.batch, device=t.device)
        return t[ar, last_idx]

    def make_hook(li):
        def hook(module, inp, out):
            last = reduce_last(out)                       # [B, E]
            idx = torch.topk(last.float(), topk, dim=-1).indices
            for e in idx.reshape(-1).tolist():
                hits[li][int(e)] += 1
            flat = out.reshape(-1, out.shape[-1])
            fidx = torch.topk(flat.float(), topk, dim=-1).indices
            for e in fidx.reshape(-1).tolist():
                allpos[li][int(e)] += 1
        return hook

    for li, layer in enumerate(model.model.layers):
        layer.mlp.gate.register_forward_hook(make_hook(li))

    with torch.no_grad():
        model(**enc)

    B, K = args.batch, topk
    useful_rows = B * K
    per_layer = []
    for li in range(cfg.num_hidden_layers):
        c = hits[li]
        activated = len(c)                      # distinct experts receiving >=1 token
        counts = list(c.values())
        processed = activated * B               # what gang_moe_linear actually computes
        ca = allpos[li]
        # UNIFIED metric: the kernel tiles M in MPerBlock=16 steps
        # (M_TILES = ceil(BATCH_SIZE/16)), so the M extent it computes is
        # ceil(B/16)*16 per activated expert -- not B.  This is the number that
        # exposes the small-batch waste.
        m_padded = -(-B // 16) * 16
        per_layer.append({
            "layer": li,
            "num_activated": activated,
            "mean_tokens_per_activated": round(statistics.mean(counts), 3),
            "median_tokens_per_activated": statistics.median(counts),
            "max_tokens_per_activated": max(counts),
            "min_tokens_per_activated": min(counts),
            "useful_rows": useful_rows,
            "processed_rows": processed,
            "tile_padded_rows": activated * m_padded,
            "row_utilisation_pct": round(100.0 * useful_rows / processed, 2),
            "row_utilisation_tile_padded_pct": round(
                100.0 * useful_rows / (activated * m_padded), 2),
            "m_tiles_per_expert": -(-B // 16),
            "utilisation_in_16row_tiles_pct": round(
                100.0 * statistics.mean(counts) / 16.0, 2),
            "allpos_num_activated": len(ca),
            "allpos_top_expert_share_pct": round(
                100.0 * max(ca.values()) / sum(ca.values()), 2),
        })

    act = [r["num_activated"] for r in per_layer]
    util = [r["row_utilisation_pct"] for r in per_layer]
    allact = [r["allpos_num_activated"] for r in per_layer]
    tokact = [r["mean_tokens_per_activated"] for r in per_layer]
    summary = {
        "model": args.model,
        "batch": B,
        "topk": K,
        "n_experts": n_experts,
        "layers": cfg.num_hidden_layers,
        "DECODE_VIEW": {
            "num_activated_mean": round(statistics.mean(act), 2),
            "num_activated_min": min(act),
            "num_activated_max": max(act),
            "mean_tokens_per_activated_expert": round(statistics.mean(tokact), 2),
            "row_utilisation_mean_pct": round(statistics.mean(util), 2),
            "row_utilisation_tile_padded_mean_pct": round(statistics.mean(
                [r["row_utilisation_tile_padded_pct"] for r in per_layer]), 2),
            "waste_factor_num_activated_over_topk": round(statistics.mean(act) / K, 2),
            "waste_factor_against_tile_padded_rows": round(
                statistics.mean([r["tile_padded_rows"] for r in per_layer])
                / useful_rows, 2),
            "MPerBlock_is_16_so_tile_fill_pct": round(
                100.0 * statistics.mean(tokact) / 16.0, 2),
        },
        "ALL_POSITIONS_VIEW": {
            "num_activated_mean": round(statistics.mean(allact), 2),
            "num_activated_min": min(allact),
            "num_activated_max": max(allact),
            "top_expert_share_mean_pct": round(statistics.mean(
                [r["allpos_top_expert_share_pct"] for r in per_layer]), 2),
        },
        "per_layer": per_layer,
    }

    print(json.dumps(summary, indent=2)[:2000])
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
