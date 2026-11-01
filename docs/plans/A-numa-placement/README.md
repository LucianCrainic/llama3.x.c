# Track A: explicit NUMA weight placement (one socket)

## Goal

The engine should own and verify where its memory lives, with no `numactl` staging
and no reliance on page-cache history. This track covers **one socket and
unchanged computation**: the same kernels, the same summation order and bitwise
identical logits. Its result is a placement backend that track B reuses for each
socket shard.

The design follows study §4 (topology), §5 (allocation contract, loader and page
verification) and §10.2 (placement acceptance).

## Phases

| ID | Phase | Output | Gate |
| --- | --- | --- | --- |
| A1 | [Topology discovery](A1-topology-discovery.md) | `numa_topology()` with socket → {CPUs, DDR node, HBM node} | hbm14 classified correctly; ambiguous layouts rejected |
| A2 | [Bound regions](A2-bound-regions.md) | `NumaRegion` alloc/bind/populate/protect/free | Bind succeeds on each node; failure modes reported clearly |
| A3 | [Placement verification](A3-placement-verification.md) | `move_pages` query, `numa_maps`/`smaps` THP check | 100% of queried pages on the requested node, 8 GiB per node |
| A4 | [Owned weight loader](A4-owned-weight-loader.md) | Checkpoint copied into bound regions; embedding in DDR | Bitwise logits vs file mapping; bounded startup memory |
| A5 | [KV and workspace placement](A5-kv-and-workspace-placement.md) | KV, activations and attention workspace in bound regions | KV pages verified after growth; bitwise logits |
| A6 | [One-socket placement campaign](A6-one-socket-placement-campaign.md) | Measured placement matrix | Engine-owned HBM ≥ staged-HBM baseline within variance |

## Track exit gate (milestone M2, placement half)

The track is done when the engine, started with `LLAMA_PLACEMENT=hbm` and pinned to
socket 0, meets all of the following:

- It decodes 8B FP32 and Q8_0 no slower than the **measured** staged baseline (14.90
  and 44.49 tok/s) beyond run variability.
- 100% of its queried weight and KV pages are on HBM node 2.
- Its report records region, placement and THP evidence.
- It prints logits identical to `LLAMA_PLACEMENT=file`.

## What track B takes from here

- `NumaRegion` and the owner-populate protocol, applied once per socket shard.
- The loader's bounded `pread` copy path. B1 adds row slicing and column packing to it.
- Verification and reporting, which B7 and D2 reuse without changes.
