# How much of a decode iteration is attention?

This matters because the A-part sweep varies **gang linear** geometry
(`GANG_TILE_N`, `GANG_M_TILES`, `GANG_WGM`). Any per-iteration cost that is common
to all configurations — attention above all — is a constant that dilutes the
*relative* differences the sweep is trying to resolve. If attention is most of the
iteration, even a large win in the linear layers moves the headline number by very
little, and the answer to question A risks being "no effect" for reasons that have
nothing to do with the parameters.

## Measurement

`demo/qwen3/demo.py --linear-only` registers only the linear tasks, skipping
attention and silu. Running it and the full pipeline with an otherwise identical
command isolates the non-linear share. Qwen3-8B, bs=32, `USE_GANG=1 USE_CK_FMHA=0
USE_FUSED_SILU=1 GANG_M_TILES_GATEUP=1 GANG_TILE_N=64`:

| run | ms/token |
|---|---|
| full (attention on) | 38.925 |
| `--linear-only` | 13.263 |

    non-linear share = (38.925 - 13.263) / 38.925 = 65.9%

So roughly **two thirds of a bs=32 decode iteration is attention**, and only about a
third is the gang linear work the sweep actually perturbs.

## Caveats

- `--linear-only` removes tasks from the graph, so it also removes their
  synchronization and scheduling overhead. The 65.9% is therefore an **upper bound**
  on attention's share, not an exact attribution.
- This is one measurement at one batch size. bs=32 is the A2 sweep point, so it is
  the relevant one, but the ratio will move with batch size and context length.
- Repeatability: pooling every configuration that was measured more than once gives a
  run-to-run spread of **median 4.0%, worst 5.9%** (`summarize.py`). An earlier
  three-sample estimate of ~2% was optimistic — do not use it. The 66/34 split is far
  outside that spread, so the share itself is solid even though the individual
  configuration timings need repeats to separate.

## What it implies for the sweep

With a 34% linear share, a genuinely large 25% win in the gang linear layers shows up
as only ~8.5% end-to-end — still detectable against a ~2% noise floor, but any
smaller effect will be buried. That is consistent with what A0 already showed:

| config | ms/token | vs default |
|---|---|---|
| `A0_tile128` | 35.349 | **-9.2%** |
| `A0_tile64` (Fleet default) | 38.925 | — |
| `A1_mpk_baseline` (no gang) | 37.913 | -2.6% |
| `A0_tile256` | 40.456 | +3.9% |

`tile_n=128` beating the default by 9.2% implies a ~27% improvement in the linear
portion — plausible, and consistent with the dilution arithmetic.

## Consequence

The A-part conclusions should be stated in terms of the **linear-portion** effect
where possible, not only the headline ms/token, and `--linear-only` should be run
alongside the main sweep so each configuration has both numbers. Otherwise a real
parameter effect can be reported as "within noise" purely because attention swamps it.

The clean fix is to stop paying for the replacement attention path at all, i.e. port
`paged_attention_minimal_decode` to gfx942 — see `MI300X-ADAPTATION-NOTES.md` §4.
