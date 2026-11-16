#!/usr/bin/env python3
"""Find the composable_kernel develop commit whose CK-tile API matches Fleet's
MI300 kernel sources.

Two markers bracket the window:
  FMHA : TileFmhaFwdSplitKVTraits must accept 14 args  -> needs kHasSink present
  GEMM : GemmPipelineAGmemBGmemCRegV2 must NOT require Policy::is_a_load_tr,
         MakeADramTensorView, is_b_load_tr, MakeBDramTensorView

We walk develop's history (newest -> oldest) and report the newest commit that
still satisfies the GEMM side while already having the FMHA side.
"""
import json
import sys
import urllib.request

REPO = "ROCm/composable_kernel"
FMHA = "include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
GEMM = "include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"


def api(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            last = e
    print("api failed", url, last, file=sys.stderr)
    return None


def raw(sha, path):
    url = f"https://raw.githubusercontent.com/{REPO}/{sha}/{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return ""


def probe(sha):
    f = raw(sha, FMHA)
    g = raw(sha, GEMM)
    return {
        "sha": sha,
        "fmha_has_sink": "kHasSink" in f,
        "gemm_policy_tr": "Policy::template is_a_load_tr" in g,
        "gemm_dramview": "MakeADramTensorView" in g,
        "gemm_ksub": "KSubTileNum" in g,
    }


def main():
    commits = []
    for page in range(1, 8):
        d = api(f"https://api.github.com/repos/{REPO}/commits?sha=develop&per_page=100&page={page}")
        if not d:
            break
        commits += [c["sha"] for c in d]
        if len(d) < 100:
            break
    print(f"develop commits fetched: {len(commits)}", file=sys.stderr)

    # binary search for the newest commit where gemm_policy_tr is False
    lo, hi = 0, len(commits) - 1   # lo = newest (index 0)
    # find first index (newest->oldest) where gemm_policy_tr is True
    first_true = None
    while lo <= hi:
        mid = (lo + hi) // 2
        p = probe(commits[mid])
        if p["gemm_policy_tr"] or p["gemm_dramview"]:
            first_true = mid
            lo = mid + 1          # older commits are at larger index
        else:
            hi = mid - 1
    print(f"newest bad (Policy-delegation present) at index {first_true}", file=sys.stderr)

    if first_true is None:
        print("no boundary found")
        return
    good = commits[first_true + 1]
    bad = commits[first_true]
    for label, sha in (("NEWEST GOOD", good), ("FIRST BAD", bad)):
        p = probe(sha)
        print(f"{label} {sha[:12]} {p}")

    # show a little context around the boundary
    print("\ncontext:")
    for i in range(max(0, first_true - 2), min(len(commits), first_true + 4)):
        p = probe(commits[i])
        print(f"  [{i}] {commits[i][:12]} sink={p['fmha_has_sink']} "
              f"policy_tr={p['gemm_policy_tr']} dramview={p['gemm_dramview']}")


if __name__ == "__main__":
    main()
