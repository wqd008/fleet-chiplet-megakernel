# Fleet on MI300X — environment & adaptation notes

Companion to `projects/fleet-chiplet-megakernel/docs/plans/mi300x-fleet-repro-and-param-search.md`.
Records what the plan assumed vs. what the machine actually is, and every change made.

## 1. Machine facts (measured, not assumed)

| Item | Plan assumed | **Actual** |
|---|---|---|
| GPU | MI300X | MI300X **VF** (virtual function), `gfx942:sramecc+:xnack-` |
| CUs | 304 | **304** ✓ |
| VRAM | 192 GB | **191.7 GB** ✓ |
| L2 / XCD | 4 MB | 4 MB (rocminfo) ✓ |
| XCDs | 8 | 8 (implied by `MAX_WORKER_PER_SCHEDULER=38`, 304/8) |
| ROCm | 7.2.4 | **7.14.1** (`/opt/rocm/core-7.14`, clang 23) |
| PyTorch | 2.10.0 preinstalled | **not installed at all** — installed `2.10.0+rocm7.0` |

Note the ROCm packaging split: system ROCm is 7.14.1, but the torch wheel bundles
HIP 7.0.51831. Fleet compiles the megakernel with the *system* `hipcc` (7.14.1),
so both toolchains are live in the same run. This works, but is worth knowing.

### Per-XCD worker arithmetic (the plan asked for this explicitly)

The megakernel prints its own launch configuration at compile time:

```
sm_cnt:  304
AMD config: workers=296, schedulers=8, physical_blocks=304, CUs=304
```

So on MI300X: 304 CUs / 8 XCDs = **38 CUs per XCD**, of which one per XCD is the
scheduler, leaving **296 workers = 37 gang workers per XCD** (the build also sets
`-DMAX_WORKER_PER_SCHEDULER=38`). On MI350 it is 248 CUs / 8 XCDs = 31 per XCD, so
~30 workers per XCD. **MI300X therefore has ~23% more workers sharing each XCD's
4 MB L2 slice than MI350 does.** That changes how much weight a single XCD must pull
through its L2 per tile, which is exactly the quantity Fleet's `GANG_TILE_N` /
`GANG_WGM` geometry trades against — and is a plausible reason the MI350-tuned
defaults do not transfer (see `results/A-part.md`).

## 2. Blockers hit and how they were resolved

The plan's §1 "前置检查" assumed the image was ready except for `AMDGPU_TARGETS`.
It was not. Five separate blockers, in the order encountered:

1. **No pip, no torch, no venv.** Resolved: `apt-get install python3-pip python3.12-venv`,
   venv at `/root/venvs/fleet`, `torch==2.10.0+rocm7.0`.
2. **Git submodules absent.** `.gitmodules` lists `deps/{cutlass,z3,json,composable_kernel}`
   but no gitlinks exist in the tree — upstream `ROCm/fleet-chiplet-megakernel` has the
   same defect (verified via the GitHub git-tree API: `deps/` contains only `rocblas`).
   Resolved by fetching manually:
   - `deps/json` @ `8c391e04fe4195d8be862c97f38cfe10e2a3472e`
   - `deps/cutlass` @ `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`
     (SHAs taken from upstream `mirage-project/mirage`, which still has its gitlinks;
     on ROCm cutlass is only needed because `setup.py:copy_include` copies its headers)
   - `deps/composable_kernel` @ `ac18460782fadcd24aa321394c63dc284c90593b` (see §3)
   - `deps/z3` not needed — `setup.py` uses the `z3-solver` wheel.
3. **`hip-lang-config.cmake` missing**, so cmake could not enable the HIP language.
   The image ships ROCm *runtime* packages but not the cmake dev files. `apt` is
   broken on this image (a pending kernel upgrade `6.8.0-149` fails its dkms
   postinst, so dpkg cannot configure anything). Resolved without apt:
   `apt-get download amdrocm-runtime-dev7.14` then `dpkg-deb -x` into `/`. This only
   adds the missing dev files; nothing else was touched.
4. **cmake 4.x rejects `deps/json`'s `cmake_minimum_required(VERSION 3.1...3.14)`.**
   Resolved by installing the same cmake the project's own CI uses (`3.27.9`).
5. **Rust toolchain** required by `setup.py` (two cargo crates) — installed via rustup.

After these, `pip install -e .` succeeds and `import mirage` works.

## 3. The composable_kernel version (the non-obvious one)

Fleet's MI300 kernels include `ck_tile` headers from `deps/composable_kernel/include`,
with the source comment "must come before system CK". That matters: the system CK
shipped with ROCm 7.14 is **API-incompatible** with Fleet's hand-written policies.

Bracketing the correct commit by probing GitHub rather than guessing:

| CK ref | FMHA `TileFmhaFwdSplitKVTraits` | GEMM pipeline needs `Policy::is_a_load_tr` | verdict |
|---|---|---|---|
| rocm-7.0.0 / 7.0.2 | 13 params, no `kHasSink` | no | too old (FMHA arity) |
| rocm-7.1.x / 7.2.x | 13 params, no `kHasSink` | no | FMHA arity mismatch |
| therock-7.9.0 / 7.10 | 13 params | no | FMHA arity mismatch |
| **`ac18460782fa`** (develop) | **14 params, has `kHasSink`** | **no** | **✓ matches** |
| next develop commit `717f2efe` | 14 | **yes** | too new |
| system ROCm 7.14 | 14 | **yes** | too new |

Fleet calls `TileFmhaFwdSplitKVTraits` with 14 template arguments (the 14th being
`kHasSink`), which pins it to a CK develop snapshot *after* sink attention landed but
*before* the pipeline policy refactor. `ac18460782fa` is the newest commit satisfying
both constraints, so that is what is checked out. (Fetched sparse — `include/` only,
114 MB.)

Consequence: **`deps/composable_kernel` must stay at that commit.** Upgrading it, or
letting `/opt/rocm/include` win the include order, breaks the build.

## 4. The one remaining MI350-ism: `USE_CK_FMHA=1`

With CK pinned, the build gets all the way down to **two** errors, both in
`include/mirage/persistent_kernel/tasks/mi300/paged_attention_decode_minimal_mi300.cuh`:

```
:27  __builtin_amdgcn_mfma_f32_16x16x32_f16   needs target feature gfx950-insts
:47  __builtin_amdgcn_ds_read_tr16_b64_v4f16  needs target feature gfx950-insts
```

Both confirmed gfx950-only by direct compile test (the 16x16x16 f16 MFMA *does*
compile for gfx942; the 16x16x32 form does not, and transposed LDS reads
(`ds_read_tr`) are CDNA4-only — they do not exist on gfx942 at all).

This file is the `seqlen_q == 1` **decode attention**, dispatched from
`paged_attention_ck_fmha_split_kv_mi300.cuh:594` — i.e. squarely on the decode
critical path, not dead code. Its name says `_mi300` but its contents are MI350.

A repo-wide scan shows this is the **only** file using gfx950-only intrinsics;
everything else (`mfma_f32_16x16x16bf16_1k`, `s_memrealtime`, `readfirstlane`,
`rcpf`, `fence`) is gfx942-safe.

**Decision taken (no kernel code modified):** run the experiment with
`USE_CK_FMHA=0`, which excludes that file via the `MPK_USE_CK_FMHA` guard in
`tasks/mi300/task_header.cuh:34`. Verified working end to end.

Why this is defensible, and its limits:
- Fleet's gang **linear** kernels — the actual subject of the A-part parameter
  search (`GANG_TILE_N`, `GANG_M_TILES`, `GANG_WGM`) — are unaffected and fully on.
  Only the *attention* backend differs.
- The same attention backend is used in every configuration, so gang-vs-non-gang
  and parameter-vs-parameter comparisons stay internally valid.
- The plan's own §3.4 already rules out comparing absolute times to the MI350 README,
  so nothing is lost there either.
- **Limitation:** attention is a per-iteration cost common to both arms. If it is a
  large fraction of the iteration, it compresses the *relative* gang speedup and
  would bias the answer to question A toward "little room". The attention share must
  therefore be quantified before drawing conclusions — see `attention_share.md`.

Alternatives considered and not taken: porting the decode kernel to gfx942 (2×
16x16x16 MFMA plus a register-level transpose to replace `ds_read_tr` — a real
attention-kernel rewrite against the plan's "do not change performance-affecting
kernel code" constraint), or routing `seqlen_q==1` to CK's prefill pipeline.

To switch later: `FLEET_CK_FMHA=1` for `param_search.py`, or `USE_CK_FMHA=1` directly.

### 4.1 The reroute was implemented and measured — it is *slower*, so it is not used

Branch `mi300x-adaptation` in the Fleet repo carries the portability fix anyway,
so that `USE_CK_FMHA=1` at least builds:

- `tasks/mi300/task_header.cuh` — includes `paged_attention_decode_minimal_mi300.cuh`
  only `#if defined(__gfx950__)`.
- `paged_attention_ck_fmha_split_kv_mi300.cuh` — on non-gfx950, `seqlen_q == 1` goes
  to `paged_attention_ck_fmha_prefill` (which takes `seqlen_q` as a runtime argument)
  instead of the MI350-only decode kernel.

Both edits sit inside `#ifdef MPK_USE_CK_FMHA` / `#if defined(__gfx950__)`, so they
are **inert when `USE_CK_FMHA=0`** — they cannot affect the numbers reported here.
Confirmed empirically as well as by inspection, since A0/A1 predate the edits and
A2–A5 postdate them: the same configurations agree within noise across the boundary
(`A0_tile64` 38.925 pre vs `A2_tile64_wgm0` 38.893 post; `A1_gang_default` 39.757 pre
vs `A4_bs32_default` 38.313 post). No systematic shift.

Measured at bs=1, same prompt, same gang config:

| config | ms/token |
|---|---|
| `USE_CK_FMHA=0` (used for the sweep) | **10.207** |
| `USE_CK_FMHA=1` (MI350 decode kernel replaced by the CK prefill pipeline) | 15.390 |

And the per-iteration profile explains why. The kernel prints `[FWD_PASS] iter=N
time_ms=...` each iteration, so the cost of one decode step can be read off directly
as the context grows:

- `USE_CK_FMHA=0`: flat — ~10.05 ms at iter 900, ~10.13 ms at iter 1100.
- `USE_CK_FMHA=1`: rising — 10.0 ms at iter 1, 18.9 ms at iter 1000, 20.7 ms at iter 1100.

The replacement pipeline re-derives attention over the whole KV range every step with
a prefill-shaped kernel, so it scales with context where Fleet's purpose-built decode
kernel would not. Substituting it is therefore a pessimisation, not a neutral swap.

**Conclusion: keep `USE_CK_FMHA=0`.** It needs no kernel modification (so it respects
the plan's §0.3 constraint), it is the faster of the two available options here, and
its attention cost is context-independent at bs=1. The genuine fix — porting
`paged_attention_minimal_decode` to gfx942 — remains open and is the single highest-
value follow-up if the attention share turns out to matter for the A-part conclusions.


## 5. Plan claims verified against source

| Plan claim | Status |
|---|---|
| `persistent_kernel.py:350` defaults `AMDGPU_TARGETS` to `gfx950` | ✓ exact |
| `persistent_kernel.py:425-427` forces `target_cc = 94` on ROCm | ✓ exact |
| gang path enabled by `assert self.target_cc == 94` | ✓ exact |
| `MPerBlock` is a compile-time 16 in the gang MoE kernel | ✓ `gang_moe_linear_mi300.cuh:73` |
| `USE_GANG_MOE` defaults to 0 | ✓ `demo_30B_A3B.py:242` |
| Qwen3-30B-A3B gate: `1536 % 64 == 0`, `2048 % 256 == 0` | ✓ (config: hidden 2048, moe_inter 768) |
| Mixtral gate: `28672 % 64 == 0`, `4096 % 256 == 0` | ✓ (config: hidden 4096, inter 14336) |
| Qwen3-8B `output_size/8` = 512 (qkv) | ✗ **actually 768** — `(32 + 2·8)·128 = 6144`, so 6144/8 = 768 |

The qkv arithmetic in the plan is wrong, but harmlessly so: the candidate set
`{32, 64, 128, 256}` is still valid for all three projections (768, 3072 and 512 are
each divisible by 256).

## 6. Two findings that change how the results must be read

### 6.1 The headline metric is not decode-only

The plan designates the `Decode: ... ms/token` line as "全部后续实验的唯一指标".
It is not a decode-only measurement. From `demo/qwen3/demo.py:1316-1326`:

```python
avg_time_per_iter = run_time / total_iterations      # prefill + decode combined
decode_time_est   = decode_iterations * avg_time_per_iter
# decode_iterations == generated_tokens, therefore
#   printed ms/token == run_time / (prefill_iterations + decode_iterations)
```

The printed value is exactly the mean wall time per megakernel iteration over
prefill **and** decode. Confirmed empirically: in the smoke run all three printed
lines read `10.207 ms` — prefill, decode and combined are literally the same number.

Consequences:
- Fine for comparing configurations **at a fixed batch size** (identical iteration mix).
- **Not** a clean TPOT, and **not** comparable across batch sizes: the prefill
  iteration count moves from ~94 at bs=1 to ~3 at bs=32, and prefill iterations cost
  far more than decode iterations. A4 (batch extension) must be read with this in mind.
- The driver therefore also records `combined_ms_per_token`, `run_tokens` and both
  iteration counts so the raw numbers stay auditable.

### 6.2 There is an undocumented `GANG_M_TILES_OVERRIDE`

`demo/qwen3/demo.py:471`:

```python
GANG_M_TILES = int(os.environ.get("GANG_M_TILES_OVERRIDE", 0)) or max(1, bs // 16)
```

The plan lists `GANG_M_TILES` as derivable only from batch size. In fact it can be set
directly at any batch size. This is used for stage A3, which lets `m_tiles` be swept
without also changing batch size (the plan's A3 relied on bs=32 making `bs//16 == 2`
coincide with the intended value).

## 7. Reproducing this environment

```bash
export AMDGPU_TARGETS=gfx942          # mandatory; in-tree default is gfx950
export MIRAGE_HOME=/root/projects/fleet-chiplet-megakernel
export MIRAGE_ROOT=$MIRAGE_HOME
export PATH=/root/venvs/fleet/bin:$PATH

cd $MIRAGE_HOME
rm -rf permanent_output_dir           # mandatory before every config change
python demo/qwen3/demo.py --use-mirage ...
```

Run matrix and driver: `param_search.py` (kept outside the Fleet tree, per the plan).
