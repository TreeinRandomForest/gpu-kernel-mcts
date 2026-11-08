# Cross-CTA reuse strategy pilot

This experiment first ran a deterministic H100 smoke search with the new
`improve_cross_cta_reuse` action. Its persisted SQLite generation prompt
contained `reuse_v1`, L2 TEX read request/hit/miss sectors, DRAM reads,
spill indicators, and explicit NCU kernel replay/`--cache-control all`
provenance. The smoke audit is in ignored `output/reuse-v1-smoke.json`.

The matched pilot then compared the frozen original four-action catalog in
[old_catalog.json](old_catalog.json) with the five-action
`configs/triton_h100_strategies.json` catalog. Each arm used the same Triton
root, BF16 GEMM workload, pinned H100 SXM image, GLM-5.3 Messages endpoint
with thinking disabled, `reuse_v1` parent profile prompt, uniform priors,
seed, and `B_gen=10`. Seed 0 ran old then new; seed 1 ran new then old.
All 40 generation calls were recorded in four ignored SQLite traces; every
generation prompt contained a `reuse_v1` parent profile.

| Seed | Four actions, best µs | Five actions, best µs | Five minus four, µs |
| ---: | ---: | ---: | ---: |
| 0 | 198.224 | 199.088 | +0.864 |
| 1 | 201.280 | 200.448 | −0.832 |

The new action was selected once across the two five-action arms; that
candidate was valid at 278.432 µs. It was not selected in seed 1. Both arms
therefore reached roughly 200 µs using other actions, and this tiny pilot
cannot establish whether the reuse action helps at a larger budget. The new
catalog changes the PUCT prior mass and action allocation, so even matched
seeds do not imply identical search trajectories.

See [analysis.json](analysis.json) for root timings, valid counts, profile
calls, and per-strategy outcomes. `python
docs/experiments/triton-h100-reuse-strategy/analyze.py` reproduces the audit
from the SQLite traces. `run.py` accepts `--budget`, `--seeds`, and `--resume`;
it uses the local `ENMAAS_BASE_URL` and `ENMAAS_KEY` without persisting either
value. The runner refuses to overwrite existing traces, archives incomplete
ones on explicit resume, and releases the Nebius worker at exit.

The later [best-node audit](best_audit.json) verified retained source and PTX
hashes. All four best sources used pointer loads and PTX `cp.async`; none used
descriptor TMA loads. Their tiles were `128×256×64` with eight warps and
grouped CTA scheduling. See the [isolated TMA source experiment](../triton-h100-tma-address/README.md)
for the subsequent address/mask test.
