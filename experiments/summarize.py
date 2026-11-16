#!/usr/bin/env python3
"""Summarize results.tsv.

Two views, because they answer different questions:

  by-config (default)
      Pools every run that used the *same* configuration, regardless of which
      stage produced it.  Identical configurations appear under several stage
      tags (A0_tile64 == A1_gang_default == A2_tile64_wgm0), and pooling them is
      what makes the ~2-6% run-to-run noise survivable.

  by-tag
      The raw view, one line per tag, so nothing is hidden by pooling.

Also reports, per configuration, whether the difference from the reference
config exceeds the observed spread -- so that "no effect" is stated as a
measurement rather than an impression.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_TSV = Path("/root/projects/fleet-exp/results/results.tsv")

# fields that define a configuration
CFG_KEYS = ["bs", "use_gang", "use_gang_moe", "tile_n", "m_tiles",
            "m_tiles_gateup", "wgm", "msplit", "ntile", "splitk",
            "ck_fmha", "fused_silu", "model_short"]


def load(path):
    with path.open() as f:
        return [r for r in csv.DictReader(f, delimiter="\t")]


def cfg_key(r):
    return tuple(r.get(k, "") or "-" for k in CFG_KEYS)


def cfg_label(k):
    d = dict(zip(CFG_KEYS, k))
    parts = [f"bs={d['bs']}"]
    parts.append("gang" if d["use_gang"] == "1" else "NOGANG")
    if d["use_gang"] == "1":
        parts.append(f"tile_n={d['tile_n']}")
        parts.append(f"mt={d['m_tiles'] or 'bs//16'}")
        parts.append(f"wgm={d['wgm'] or '0'}")
        if d["m_tiles_gateup"] not in ("", "-"):
            parts.append(f"mtgu={d['m_tiles_gateup']}")
        for f, lbl in (("msplit", "msplit"), ("ntile", "ntile"),
                       ("splitk", "splitk")):
            if d[f] == "1":
                parts.append(lbl)
    parts.append(f"ckfmha={d['ck_fmha']}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=str(DEFAULT_TSV))
    ap.add_argument("--stage", default=None)
    ap.add_argument("--baseline", default=None,
                    help="tag or config label to compare against; default = best")
    args = ap.parse_args()

    rows = load(Path(args.tsv))
    if args.stage:
        rows = [r for r in rows if r["stage"] == args.stage]
    if not rows:
        print("no rows")
        return

    cfg_vals = defaultdict(list)
    tag_vals = defaultdict(list)
    for r in rows:
        if r.get("ms_per_token"):
            v = float(r["ms_per_token"])
            cfg_vals[cfg_key(r)].append(v)
            tag_vals[r["tag"]].append(v)

    def stats(vals):
        med = statistics.median(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return med, sd, min(vals), max(vals)

    # ---- by config ----
    print("=" * 96)
    print("BY CONFIGURATION (pooled across stages)")
    print("=" * 96)
    print(f"{'configuration':<48} {'n':>2} {'median':>8} {'sd':>7} {'spread%':>8}")
    print("-" * 96)
    ranked = sorted(cfg_vals.items(), key=lambda kv: statistics.median(kv[1]))
    # "vs best" must be computed WITHIN a batch size: ms/token is not comparable
    # across batch sizes (see attention_share.md / adaptation notes 6.1).
    best_by_bs = {}
    for k, vals in ranked:
        bs = dict(zip(CFG_KEYS, k))["bs"]
        med = statistics.median(vals)
        best_by_bs[bs] = min(best_by_bs.get(bs, med), med)

    for k, vals in ranked:
        bs = dict(zip(CFG_KEYS, k))["bs"]
        med, sd, lo, hi = stats(vals)
        spread = (hi - lo) / med * 100 if len(vals) > 1 else float("nan")
        rel = (best_by_bs[bs] - med) / best_by_bs[bs] * 100
        sp = f"{spread:7.1f}%" if len(vals) > 1 else "      -"
        print(f"{cfg_label(k):<48} {len(vals):>2} {med:>8.3f} {sd:>7.3f} {sp:>8}  "
              f"{rel:+.1f}% vs best@bs{bs}")

    # noise floor from configs measured more than once
    repeats = [v for v in cfg_vals.values() if len(v) > 1]
    if repeats:
        floors = [(max(v) - min(v)) / statistics.median(v) * 100 for v in repeats]
        print(f"\nobserved run-to-run spread on repeated configs: "
              f"median {statistics.median(floors):.1f}%, worst {max(floors):.1f}% "
              f"(n={len(floors)} configs)")
        print("=> differences smaller than this are not resolvable at this repeat count; "
              "add repeats before calling them real.")

    # ---- by tag ----
    print()
    print("=" * 96)
    print("BY TAG (unpooled; identical configs may appear under several tags)")
    print("=" * 96)
    print(f"{'tag':<26} {'n':>2} {'median':>8} {'min':>8} {'max':>8} {'spread%':>8}")
    print("-" * 96)
    for tag, vals in sorted(tag_vals.items(), key=lambda kv: statistics.median(kv[1])):
        med, sd, lo, hi = stats(vals)
        spread = (hi - lo) / med * 100 if len(vals) > 1 else float("nan")
        sp = f"{spread:7.1f}%" if len(vals) > 1 else "      -"
        print(f"{tag:<26} {len(vals):>2} {med:>8.3f} {lo:>8.3f} {hi:>8.3f} {sp:>8}")

    # ---- failures ----
    failed = [r for r in rows if not r.get("ms_per_token") or r.get("compile_ok") == "0"]
    if failed:
        print("\nconfigs with no timing line (kept on purpose -- 'which values are")
        print("unusable' is itself a result):")
        for r in failed:
            print(f"  {r['tag']:<26} exit={r['exit_code']:>4} "
                  f"timeout={r['timed_out']} {r['note'][:90]}")


if __name__ == "__main__":
    main()
