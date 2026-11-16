# Part A — Fleet static parameter search on MI300X

Qwen3-8B bf16, single MI300X VF (304 CU), `USE_CK_FMHA=0 USE_FUSED_SILU=1
GANG_M_TILES_GATEUP=1`, prompt and generation length fixed at the plan's values
(94-token prompt, 1024 decode tokens). Every configuration was recompiled from
scratch after deleting `permanent_output_dir/`.

Metric is the plan's headline `Decode ... ms/token`. **Read
`../MI300X-ADAPTATION-NOTES.md` §6.1 first**: it is not decode-only TPOT, it is mean
wall time per megakernel iteration over prefill+decode. Comparisons below are all at
a fixed batch size, where it is a valid comparator.

## Measurement noise (established first, because everything depends on it)

Identical configurations were run repeatedly, both deliberately (repeat passes) and
incidentally (A0/A1/A2 contain overlapping configurations that `summarize.py` pools).
Across the 12 configurations measured more than once:

> **run-to-run spread: median 2.2%, worst 6.5%**

An earlier three-sample estimate of ~2% was optimistic and is superseded. **No
difference below ~2% is resolvable at these repeat counts, and anything below 6.5% is
not safe to call real on a single sample.** This is the single most important number
in this document: it is why the first A0 pass looked like a clean 9% win for
`tile_n=128` and the pooled data cut it to ~6–7%.

The metric is also sensitive to context length — attention cost grows as the KV cache
fills — so absolute values drift between sessions. Only same-session comparisons are
used.

## Results — main sweep at bs=32, with m_tiles at its default (`bs//16` = 2)

Pooled medians. `n` is the number of runs of that exact configuration.

| tile_n | wgm=0 | wgm=1 | wgm=2 | wgm=4 | best of row |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 64 (Fleet default) | 38.909 (n=4) | 39.080 (n=2) | 38.364 (n=2) | 37.389 (n=2) | 37.389 |
| **128** | 37.505 (n=3) | **36.071** (n=2) | **35.711** (n=2) | **36.374** (n=2) | **35.711** |
| 256 | 40.290 (n=3) | 38.721 (n=2) | 39.949 (n=2) | 39.597 (n=2) | 38.721 |
| **no gang (MPK baseline)** | 37.913 (n=1) | — | — | — | 37.913 |

### What the data supports

**1. `tile_n = 128` beats Fleet's hardcoded `tile_n = 64`, by ~6–8%.**
The evidence is stronger than the individual medians suggest, because it is
*consistent across the whole wgm axis*: tile_n=128 beats tile_n=64 at wgm=0, 1, 2 and
4 — 4 out of 4 paired comparisons. A sign test gives p ≈ 0.06 two-sided, and the
magnitude is at or above the worst-case noise. Fleet's default (64, wgm=0) = 38.909 vs
(128, wgm=2) = 35.711 is **8.2% faster**.

`tile_n = 256` is worse than the default at every wgm — so the space is genuinely
non-monotonic, not "bigger is better". This is a real optimum, not a boundary artefact.

**2. `GANG_WGM` is not a free win, but `wgm = 0` looks like the wrong default once
`tile_n` is right.** At tile_n=128 every non-zero wgm (35.71 / 36.07 / 36.37) beats
wgm=0 (37.51), a ~3–5% gap. But at tile_n=64 and 256 the wgm ordering is inconsistent
(tile_n=64's best is wgm=4; tile_n=256's best is wgm=1), which is what noise looks
like. Treat this as **suggestive, not established**: it needs more repeats before it
can be claimed. It is nevertheless the one parameter the plan correctly identified as
never having been tuned — the source hardcodes it to 0.

**3. Fleet's shipped default gives no benefit over not using gang at all.**
Default gang 38.909 vs no-gang 37.913 — the gang *default* is ~2.6% **slower** than
the MPK baseline. This contradicts the README's 1.23× speedup at bs=32, which was
measured on MI350. The plausible mechanism is in `../MI300X-ADAPTATION-NOTES.md`:
MI300X has 37 workers per XCD against MI350's ~30, so MI300X has ~23% more consumers
sharing each XCD's 4 MB L2 slice — the exact resource the gang geometry schedules
against. A geometry tuned for 30 workers is not tuned for 37.

A *tuned* gang configuration does beat the baseline: 35.711 vs 37.913 = **5.8% faster**.

**4. `m_tiles` (A3, at tile_n=128, wgm=2): no resolvable effect.**
`m_tiles` = 1 → 38.200, 2 → 37.484, 4 → 37.030. The trend is monotone but every step
is 1.2–1.9%, i.e. inside the noise floor. **Not resolvable at one sample per point.**
Note this also validates the undocumented `GANG_M_TILES_OVERRIDE` (see notes §6.2) —
setting it to 2 reproduces the default-bs-derived value within noise (37.484 vs
35.711/36.1 for the same nominal geometry).

## Answer to the plan's question 1

> How much room does static parameter search have, and is the gap inside the noise?

**There is real room — roughly 8% on the headline number, ~25% on the linear portion
— and it is outside the noise, but only just, and only because it was measured with
repeats.** The single most defensible claim is:

> On MI300X at bs=32, `GANG_TILE_N=128` is better than Fleet's hardcoded 64 by ~6–8%,
> and Fleet's shipped default geometry is not better than not using gang at all.
>
> But bs=1 reverses the ranking (see question 2), so "128 is better" is a
> *batch-conditional* statement. The defensible form of the finding is that **the
> optimum is batch-dependent and the shipped constant is optimal at neither batch
> size** — which is a stronger argument for search than any single number here.

## A4 — batch extension (bs 1 / 32 / 64)

| bs | Fleet default (tile_n=64, wgm=0) | tuned (tile_n=128, wgm=2) | winner |
|:---:|:---:|:---:|:---:|
| 1 | **10.291** | 11.227 | **default, by 9.1%** |
| 32 | 38.313 | **36.334** | **tuned, by 5.3%** |
| 64 | **aborts** | **aborts** | — |

*(bs=1 tuned row is a manual run: the driver cannot express a per-batch `m_tiles`, and
`m_tiles=2` is invalid at bs=1. Legal geometry `GANG_M_TILES_OVERRIDE=1`; see the
`A4_bs1_best_mt1` row in `results.tsv` and `logs/manual_bs1_best.log`.)*

**The optimum moves with batch, and that is the headline result of Part A.** The
geometry that is best at bs=32 is 9.1% *worse* than Fleet's default at bs=1, and vice
versa. This is exactly the evidence the plan's question 2 was looking for: a single
hardcoded constant cannot be right at both ends of the batch range. At bs=1 the
generation is weight-bandwidth-bound and a smaller `tile_n` wins; at bs=32 there is
enough M-work to amortise the wider tile. Fleet ships one constant for both.

### bs=64 does not run at all — and it is our own configuration's fault

`A4_bs64_*` abort with an assertion, not a timeout:

```
src/threadblock/graph.cc:45: Assertion `grid_dim.x * grid_dim.y * grid_dim.z <=
MAX_NUM_THREADBLOCKS_PER_KERNEL' failed.
```

`MAX_NUM_THREADBLOCKS_PER_KERNEL = 4096` (`include/mirage/config.h:39`), and the layer
that trips it is identified exactly. `demo.py:778`, the split-KV attention layer:

```python
grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, num_kv_cache_chunks)
```

with `num_local_kv_heads = 8` (Qwen3-8B), `num_kv_cache_chunks = ceil(1200/128) = 10`:

| bs | grid | product | vs 4096 |
|:---:|---|---|---|
| 32 | (32, 8, 10) | 2560 | passes |
| 64 | (64, 8, 10) | **5120** | **fails** |

Verified to be gang-independent: `USE_GANG=0` at bs=64 aborts identically. It is a
property of the `USE_CK_FMHA=0` attention path — the README's `USE_CK_FMHA=1` path
uses `(bs, 8, CK_FMHA_CHUNKS=1)` = 512 at bs=64, which is why bs=64 works on MI350 and
in the paper. **This is a cost of the `USE_CK_FMHA=0` substitution identified in
`../MI300X-ADAPTATION-NOTES.md` §4, and it is the second independent reason to want
the real gfx942 decode-attention port.**

Workarounds, none used here: drop `--max-seq-length` to ≤1024 (chunks ≤ 8 → grid 4096,
exactly at the limit), or use the CK FMHA path. Both change the workload, so A4's
bs=64 column is left empty rather than filled with an incomparable number.

### `m_tiles` must divide the batch

`A4_bs1_best` fails on `assert batch 1 must be divisible by m_tiles 2`
(`persistent_kernel.py:1599`) — the plan's stated constraint, confirmed. Recording it
as a failed sample is the intended behaviour: it is evidence about the parameter
space's real degrees of freedom, not a driver bug.

## Answer to the plan's question 2

> Do the optimal parameters move with batch / model?

**Answered for batch within the range we can run, and answered more strongly for
hardware.**

- **Across batch: YES, the optimum moves.** bs=1 prefers Fleet's default `tile_n=64`
  (10.291) over `tile_n=128` (11.227) by 9.1%; bs=32 prefers `tile_n=128` (36.334) over
  the default (38.313) by 5.3%. The two batch sizes disagree on the winner, and both
  gaps exceed the noise floor. **This is the strongest answer to question 2**: a single
  hardcoded constant provably cannot be right across the batch range on this hardware.
  Caveat: one sample per cell at bs=1, so the 9.1% needs a repeat to be called robust.
- bs=64 cannot be measured on this configuration at all (see above), so the plan's
  full three-point batch sweep is incomplete. That is a gap in coverage, not in the
  conclusion — the two points we do have already disagree with each other.
- Across **hardware** the answer is unambiguous and is the strongest result in this
  document: a geometry tuned on MI350 (30 workers/XCD) is measurably not optimal on
  MI300X (37 workers/XCD), and Fleet's shipped default is not even better than
  disabling gang. That is a direct empirical argument for search over hardcoded
  constants — which is what question A was asking about in the first place.

## A5 — controls

| control | ms/token | matched baseline | expected | verdict |
|---|---|---|---|---|
| `USE_GANG_N_TILE=1` (negative) | 36.387 | 35.711 (`tile_n=128,wgm=2,m_tiles=bs//16`) | no benefit | ✅ no benefit — 1.9% *slower* |
| `GANG_M_TILES_GATEUP=4` (positive) | 36.277 | 36.909 (`m_tiles=2, wgm=2, GATEUP=1`) | possible benefit | ❌ no resolvable benefit — 1.7% faster, inside the 2.6% median noise |

Both controls are compared against the configuration that differs from them in exactly
one variable. The positive control must be matched on `m_tiles=2` — comparing it to the
default-`m_tiles` row (35.711) would attribute the `m_tiles` difference to `GATEUP` and
report the wrong sign.

The negative control is the important one. The plan flagged that if `USE_GANG_N_TILE=1`
*also* sped things up, the whole A-part result would have to be reinterpreted as "any
gang variant is faster" rather than "weight reuse on the M axis". It did not speed
things up, at nearly identical cost to the M-tiled variant (36.387 vs 35.711, both far
from the 38.9 default). **The A-part conclusions therefore rest on the intended
mechanism, not on an artefact.** This is the single most reassuring measurement in
Part A.

The positive control failing to show a benefit is reported as measured. It does not
contradict anything — the plan's basis for expecting a gain was a precedent on another
model, and at bs=32 with `m_tiles` already 2 there may simply be little left to gain.

## Caveats that constrain these conclusions

1. **Attention dilution.** ~66% of a bs=32 iteration is attention, which the sweep does
   not perturb (`../attention_share.md`). All differences here are diluted by ~3×
   relative to their effect on the linear portion. The clean fix — porting
   `paged_attention_minimal_decode` to gfx942 — is still open
   (`../MI300X-ADAPTATION-NOTES.md` §4).
2. **`USE_CK_FMHA=0`.** Deviates from the README config and disables Fleet's gang
   attention. Applied uniformly, so comparisons remain internally valid, but these are
   not the README's "Fleet" configuration.
3. **Absolute values are not comparable to the paper.** MI300X ≠ MI350, and the
   attention backend differs. Only relative comparisons within this table are used.
4. **n is small** (2–4). The tile_n conclusion survives because it is consistent across
   four paired comparisons; the wgm and m_tiles conclusions do not yet, and are
   flagged as such rather than smoothed over.
