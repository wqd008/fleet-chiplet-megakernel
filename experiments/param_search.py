#!/usr/bin/env python3
"""Fleet static parameter search driver (MI300X).

Runs demo/qwen3/demo.py under a given env/CLI configuration, parses the timing
lines, and appends one row per run to a TSV.  Does NOT modify Fleet's kernel or
python code -- every knob goes through the existing env-var channel.

Usage:
    python param_search.py --stage A0 --repeat 1 --model Qwen/Qwen3-8B
    python param_search.py --stage A3 --tile-n 128 --wgm 2
    python param_search.py --dry-run --stage A2

Design notes / caveats
----------------------
* `permanent_output_dir/` MUST be deleted before every run: the megakernel is
  compiled per (flags x shapes) mode and a stale dir silently reuses old
  artifacts.
* AMDGPU_TARGETS=gfx942 is mandatory; the in-tree default is gfx950 (MI350).
  Forgetting it produces a binary that cannot run on MI300X.
* The reported "Decode ... ms/token" is NOT a decode-only measurement.  From
  demo/qwen3/demo.py:1316-1326:
        avg_time_per_iter = run_time / total_iterations
        decode_time_est   = decode_iterations * avg_time_per_iter
  and decode_iterations == generated_tokens, so the printed ms/token is exactly
  `run_time / (prefill_iterations + decode_iterations)`, i.e. the mean wall time
  per megakernel iteration over prefill+decode combined.  It is a consistent
  comparator at a fixed batch size but is not a clean TPOT, and is NOT directly
  comparable across batch sizes (the prefill/decode iteration mix changes).
  We therefore also record `combined_ms_per_token`, `run_tokens` and the
  iteration counts so the raw numbers stay auditable.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FLEET_ROOT = Path(os.environ.get("FLEET_ROOT", "/root/projects/fleet-chiplet-megakernel"))
FLEET_PYTHON = os.environ.get("FLEET_PYTHON", "/root/venvs/fleet/bin/python")
RESULTS_DIR = Path(os.environ.get("FLEET_RESULTS", "/root/projects/fleet-exp/results"))
LOG_DIR = Path(os.environ.get("FLEET_LOGS", "/root/projects/fleet-exp/logs"))

# The canonical prompt from the Fleet README / the experiment plan.  Keeping it
# byte-identical across every configuration is what makes the runs comparable
# (it fixes prompt_len, hence the prefill iteration count).
PROMPT = (
    "Explain the key differences between RISC and CISC processor architectures, "
    "including their instruction set design philosophies, pipeline implementations, "
    "memory access patterns, register file organization, microcode usage, and how "
    "modern processors have evolved to blur the traditional boundaries between "
    "these two approaches in terms of performance, power efficiency, and overall "
    "transistor utilization."
)

DECODE_RE = re.compile(
    r"Decode:\s*(\d+)\s*tokens in\s*(\d+)\s*iterations\s*[≈~]\s*([\d.]+)\s*ms total\s*\(([\d.]+)\s*ms/token\)"
)
PREFILL_RE = re.compile(
    r"Prefill:\s*(\d+)\s*tokens in\s*(\d+)\s*iterations\s*[≈~]\s*([\d.]+)\s*ms total\s*\(([\d.]+)\s*ms/token\)"
)
COMBINED_RE = re.compile(
    r"Combined:\s*(\d+)\s*tokens, per-token latency:\s*([\d.]+)\s*ms"
)

TSV_COLUMNS = [
    "run_id", "tag", "stage", "repeat", "model_short", "bs", "tile_n", "m_tiles",
    "m_tiles_gateup", "wgm", "msplit", "ntile", "splitk", "ksplits",
    "use_gang", "use_gang_moe", "ck_fmha", "fused_silu",
    "ms_per_token", "combined_ms_per_token", "prefill_ms_per_token",
    "decode_iters", "prefill_iters", "run_tokens",
    "seconds", "exit_code", "compile_ok", "timed_out", "note",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def base_env(extra):
    env = dict(os.environ)
    env["AMDGPU_TARGETS"] = "gfx942"          # MI300X; in-tree default is gfx950
    env["MIRAGE_HOME"] = str(FLEET_ROOT)
    env["MIRAGE_ROOT"] = str(FLEET_ROOT)
    env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    # Keep the on-GPU timing path stable across runs.
    env.pop("MPK_ENABLE_DEVICE_TASK_TIMING", None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def make_cfg(tag, stage, env, args, model, note="", repeat=0):
    return {
        "tag": tag, "stage": stage, "env": dict(env), "args": list(args),
        "model": model, "note": note, "repeat": repeat,
    }


def bs_args(bs, max_new_tokens=1024, max_seq_length=1200):
    return [
        "--max-num-batched-tokens", str(bs),
        "--max-num-batched-requests", str(bs),
        "--max-seq-length", str(max_seq_length),
        "--max-new-tokens", str(max_new_tokens),
        "--ignore-eos",
        "--prompt", PROMPT,
    ]


# USE_CK_FMHA: the README/plan config uses 1, but on MI300X the CK-FMHA path is
# blocked by gfx950-only intrinsics in paged_attention_decode_minimal_mi300.cuh
# (see docs/notes in results).  Default to 0 so every config shares one attention
# backend; override with FLEET_CK_FMHA=1 once that kernel is ported.
CK_FMHA = os.environ.get("FLEET_CK_FMHA", "0")


def gang_env(tile_n=None, m_tiles=None, m_tiles_gateup=None, wgm=None,
             msplit=None, ntile=None, splitk=None, ksplits=None,
             use_gang="1", fused_silu="1", ck_fmha=None):
    e = {"USE_GANG": use_gang, "USE_CK_FMHA": CK_FMHA if ck_fmha is None else ck_fmha,
         "USE_FUSED_SILU": fused_silu}
    if tile_n is not None:
        e["GANG_TILE_N"] = str(tile_n)
    if wgm is not None:
        e["GANG_WGM"] = str(wgm)
    # NOTE: GANG_M_TILES_OVERRIDE is undocumented in the plan but honoured at
    # demo/qwen3/demo.py:471 -- it lets us set m_tiles independently of bs.
    if m_tiles is not None:
        e["GANG_M_TILES_OVERRIDE"] = str(m_tiles)
    if m_tiles_gateup is not None:
        e["GANG_M_TILES_GATEUP"] = str(m_tiles_gateup)
    if msplit is not None:
        e["USE_GANG_M_SPLIT"] = str(msplit)
    if ntile is not None:
        e["USE_GANG_N_TILE"] = str(ntile)
    if splitk is not None:
        e["USE_GANG_SPLITK"] = str(splitk)
    if ksplits is not None:
        e["GANG_K_SPLITS"] = str(ksplits)
    return e


# Fleet's shipped configuration, exactly as the README describes it.
FLEET_DEFAULT_ENV = gang_env(tile_n=64, wgm=0, m_tiles_gateup=1)
MIRAGE_ENV = {"USE_GANG": "0", "USE_CK_FMHA": CK_FMHA, "USE_FUSED_SILU": "1"}


def build_stages(model, tile_n=None, wgm=None, m_tiles=None):
    """Return {stage_name: [cfg, ...]} following the plan's staged ladder."""
    stages = {}

    # ---- A0: feasibility.  Which tile_n values even compile/run? ----
    stages["A0"] = [
        make_cfg(f"A0_tile{tn}", "A0", gang_env(tile_n=tn, wgm=0, m_tiles_gateup=1),
                 bs_args(32), model,
                 note="feasibility probe (defaults elsewhere)")
        for tn in (64, 128, 256)
    ]

    # ---- A1: anchors.  MPK baseline vs Fleet default ----
    stages["A1"] = [
        make_cfg("A1_mpk_baseline", "A1", MIRAGE_ENV, bs_args(32), model,
                 note="USE_GANG=0 (Mirage/MPK baseline)"),
        make_cfg("A1_gang_default", "A1", FLEET_DEFAULT_ENV, bs_args(32), model,
                 note="Fleet shipped default (tile_n=64,wgm=0,m_tiles=bs//16)"),
    ]

    # ---- A2: main sweep, tile_n x wgm at bs=32 (m_tiles defaults to 2) ----
    stages["A2"] = [
        make_cfg(f"A2_tile{tn}_wgm{w}", "A2",
                 gang_env(tile_n=tn, wgm=w, m_tiles_gateup=1),
                 bs_args(32), model, note="main sweep bs=32 m_tiles=2")
        for tn in (64, 128, 256) for w in (0, 1, 2, 4)
    ]

    # ---- A3: second face, m_tiles at the A2 winner (supplied on CLI) ----
    tn = tile_n if tile_n is not None else 64
    w = wgm if wgm is not None else 0
    stages["A3"] = [
        make_cfg(f"A3_tile{tn}_wgm{w}_mt{m}", "A3",
                 gang_env(tile_n=tn, wgm=w, m_tiles=m, m_tiles_gateup=1),
                 bs_args(32), model,
                 note=f"m_tiles sweep (tile_n={tn},wgm={w}); via GANG_M_TILES_OVERRIDE")
        for m in (1, 2, 4)
    ]

    # ---- A4: does the optimum move with batch? ----
    best_env = gang_env(tile_n=tn, wgm=w,
                        m_tiles=m_tiles, m_tiles_gateup=1)
    stages["A4"] = [
        make_cfg(f"A4_bs{bs}_default", "A4", FLEET_DEFAULT_ENV, bs_args(bs), model,
                 note="Fleet default at this bs")
        for bs in (1, 32, 64)
    ] + [
        make_cfg(f"A4_bs{bs}_best", "A4", best_env, bs_args(bs), model,
                 note=f"A2/A3 winner (tile_n={tn},wgm={w},m_tiles={m_tiles} or default)")
        for bs in (1, 32, 64)
    ]

    # ---- A5: positive/negative controls ----
    stages["A5"] = [
        make_cfg(f"A5_ntile_tile{tn}_wgm{w}", "A5",
                 gang_env(tile_n=tn, wgm=w, ntile=1, m_tiles_gateup=1),
                 bs_args(32), model,
                 note="NEGATIVE control: USE_GANG_N_TILE=1 (expect no weight-reuse win)"),
        make_cfg(f"A5_gateup_gt_mt_tile{tn}_wgm{w}", "A5",
                 gang_env(tile_n=tn, wgm=w, m_tiles=m_tiles, m_tiles_gateup=4),
                 bs_args(32), model,
                 note="POSITIVE control: GANG_M_TILES_GATEUP != GANG_M_TILES"),
    ]

    return stages


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def run_one(cfg, timeout=3600, run_id="", dry_run=False, extra_label=""):
    tag = cfg["tag"] + (f"__{extra_label}" if extra_label else "")
    env = base_env(cfg["env"])
    cmd = [FLEET_PYTHON, "demo/qwen3/demo.py", "--use-mirage"] + cfg["args"]
    if cfg["model"]:
        cmd += ["--model", cfg["model"]]

    row = {c: "" for c in TSV_COLUMNS}
    row.update({
        "run_id": run_id, "tag": tag, "stage": cfg["stage"],
        "repeat": cfg["repeat"], "note": cfg["note"],
        "model_short": (cfg["model"] or "").split("/")[-1],
    })
    for key, col in [
        ("GANG_TILE_N", "tile_n"), ("USE_GANG", "use_gang"),
        ("USE_GANG_MOE", "use_gang_moe"), ("GANG_WGM", "wgm"),
        ("USE_GANG_M_SPLIT", "msplit"), ("USE_GANG_N_TILE", "ntile"),
        ("USE_GANG_SPLITK", "splitk"), ("GANG_K_SPLITS", "ksplits"),
        ("USE_CK_FMHA", "ck_fmha"), ("USE_FUSED_SILU", "fused_silu"),
        ("GANG_M_TILES_OVERRIDE", "m_tiles"), ("GANG_M_TILES_GATEUP", "m_tiles_gateup"),
    ]:
        row[col] = cfg["env"].get(key, "")
    # batch size lives in the CLI args
    try:
        i = cfg["args"].index("--max-num-batched-tokens")
        row["bs"] = cfg["args"][i + 1]
    except ValueError:
        pass

    if dry_run:
        print(f"[dry-run] {tag}\n    env: " +
              " ".join(f"{k}={v}" for k, v in sorted(cfg["env"].items())) +
              "\n    cmd: " + " ".join(cmd))
        return None

    # 1. stale megakernel must go
    shutil.rmtree(FLEET_ROOT / "permanent_output_dir", ignore_errors=True)

    log_path = LOG_DIR / f"{run_id}__{tag}.log"
    t0 = time.time()
    timed_out = False
    try:
        # 3. run
        p = subprocess.run(cmd, cwd=str(FLEET_ROOT), env=env, timeout=timeout,
                           capture_output=True, text=True, errors="replace")
        out, err, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        code = -1
    seconds = round(time.time() - t0, 1)

    combined = out + "\n" + err
    log_path.write_text(combined, errors="replace")

    # 4. parse
    m = DECODE_RE.search(combined)
    if m:
        row["decode_iters"] = m.group(2)
        row["ms_per_token"] = m.group(4)
    pm = PREFILL_RE.search(combined)
    if pm:
        row["prefill_iters"] = pm.group(2)
        row["prefill_ms_per_token"] = pm.group(4)
    cm = COMBINED_RE.search(combined)
    if cm:
        row["run_tokens"] = cm.group(1)
        row["combined_ms_per_token"] = cm.group(2)

    # 5. status
    row["seconds"] = seconds
    row["exit_code"] = code
    row["timed_out"] = int(timed_out)
    row["compile_ok"] = int(m is not None)
    if not row["note"]:
        row["note"] = ""
    if timed_out:
        row["note"] = (row["note"] + " | TIMEOUT").strip(" |")
    elif m is None:
        tail = [l for l in combined.strip().splitlines() if l.strip()][-1:] or [""]
        row["note"] = (row["note"] + " | no timing line; last: " + tail[0][:160]).strip(" |")

    return row


def append_tsv(row, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if new:
            f.write("\t".join(TSV_COLUMNS) + "\n")
        f.write("\t".join(str(row.get(c, "")) for c in TSV_COLUMNS) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="A0",
                    help="comma-separated stages: A0,A1,A2,A3,A4,A5 (default A0)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tile-n", type=int, default=None, help="A2/A3 winner tile_n")
    ap.add_argument("--wgm", type=int, default=None, help="A2 winner GANG_WGM")
    ap.add_argument("--m-tiles", type=int, default=None, help="A4 winner m_tiles")
    ap.add_argument("--repeat", type=int, default=1, help="repeats per config (median taken later)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-config timeout, seconds")
    ap.add_argument("--out", default=None, help="results TSV path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-consecutive-failures", type=int, default=3,
                    help="abort the whole stage after this many compile failures")
    ap.add_argument("--label", default="", help="extra suffix appended to tags")
    args = ap.parse_args()

    stages = build_stages(args.model, args.tile_n, args.wgm, args.m_tiles)
    wanted = [s.strip() for s in args.stage.split(",") if s.strip()]
    for s in wanted:
        if s not in stages:
            sys.exit(f"unknown stage {s!r}; known: {sorted(stages)}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else RESULTS_DIR / "results.tsv"

    total = sum(len(stages[s]) for s in wanted) * args.repeat
    print(f"run_id={run_id}  stages={wanted}  configs={total}  out={out_path}")
    print(f"FLEET_ROOT={FLEET_ROOT}  python={FLEET_PYTHON}")

    consec_fail = 0
    done = 0
    for s in wanted:
        for cfg in stages[s]:
            for rep in range(args.repeat):
                c = dict(cfg)
                c["repeat"] = rep
                done += 1
                print(f"\n[{done}/{total}] {c['tag']} (rep {rep}) ...", flush=True)
                row = run_one(c, timeout=args.timeout, run_id=run_id,
                              dry_run=args.dry_run, extra_label=args.label)
                if row is None:
                    continue
                append_tsv(row, out_path)
                print(f"    ms/token={row['ms_per_token'] or 'FAIL'}  "
                      f"exit={row['exit_code']}  {row['seconds']}s", flush=True)
                if not row["compile_ok"]:
                    consec_fail += 1
                    if consec_fail >= args.max_consecutive_failures:
                        sys.exit(f"ABORT: {consec_fail} consecutive configs produced no "
                                 f"timing line. Stopping stage {s}.")
                else:
                    consec_fail = 0

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
