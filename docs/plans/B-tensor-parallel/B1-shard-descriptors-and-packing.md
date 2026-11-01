# B1. Shard descriptors and packing

## 1. Goal and status

Build each socket's weight shard from the validated manifest, directly into that
socket's bound region, and populate it with that socket's team.

- **Output-row partitions** are contiguous row copies.
- **Input-column partitions** (Wo, W2) are packed through bounded DDR scratch.
- **Q8 scales** travel with their values and stay aligned to their groups.

Correctness is shown by reconstructing every original tensor exactly. B1 changes no
computation.

Status: **planned**. Branch: `feature/tp-b1-shards`.

## 2. Depends on / unblocks

- Depends on: F1 (manifest) and A4 (owned loader, regions, parallel `pread`).
- Unblocks: B2. It also defines the descriptors that C5, C6 and D1 pack AMX layouts
  from.

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| Manifest | F1 `TensorDesc` | Rows, columns, offsets and scale offsets |
| Loader | A4 | Region per {socket, role}; parallel `pread`; owner-team population; verification |
| Q8 layout | `src/runq.c:136-149` | Each tensor stores `q[rows·cols]`, then `s[rows·cols/gs]`. Row r's scales are `s + r·(cols/gs)` (`src/kernels_simd.c:506-508`). |
| Group checks | `src/runtime.c:100-120` | `gs` divides `dim` and `hidden_dim` |

## 4. Changes

### 4.1 Partition table

Generic form of study §6.1. `S` is the socket count, 1 or 2:

| Tensor | Full `[out][in]` | Socket s shard | Kind |
| --- | --- | --- | --- |
| Wq | `dim × dim` | rows `[s·dim/S, (s+1)·dim/S)` | `ROWS` |
| Wk, Wv | `kv_dim × dim` | rows `[s·kv_dim/S, …)` | `ROWS` |
| Wo | `dim × dim` | columns `[s·dim/S, …)`, all rows | `COLS` (packed) |
| W1, W3 | `hidden × dim` | rows `[s·hidden/S, …)` | `ROWS` |
| W2 | `dim × hidden` | columns `[s·hidden/S, …)` | `COLS` (packed) |
| Classifier | `vocab × dim` | rows `[s·vocab/S, …)` | `ROWS` |
| Norm weights | `dim` | full copy per socket | `COPY` |
| Untied embedding | `vocab × dim` | one copy in the coordinator socket's DDR | `GLOBAL` |
| Tied embedding/classifier | `vocab × dim` | vocabulary-row shards, owned per socket; the coordinator fetches an embedding row from its owner | `ROWS` + `alias` |

### 4.2 Validation, in addition to the F1 checks

- `n_kv_heads % S == 0`, so complete KV heads are owned.
- `n_heads / n_kv_heads` (GQA) groups stay complete. The query-head ranges must equal
  `kv_mul ×` the KV-head ranges.
- `hidden % S == 0`.
- `vocab` may be odd. For example, 128256 / 2 = 64128 is even, but the general case
  gives the remainder to the last socket.
- **Q8.** `(dim/S) % gs == 0` and `(hidden/S) % gs == 0`, i.e. `2048 % gs` and
  `7168 % gs` for 8B. With `gs = 64`, both hold.
- **Incompatible group sizes are rejected** with a clear error. Never re-quantize
  (study §5.4).

### 4.3 Descriptor

```c
/* Proposed, include/shards.h. */
typedef enum { SHARD_ROWS, SHARD_COLS, SHARD_COPY, SHARD_GLOBAL } ShardKind;
typedef struct {
  const TensorDesc *src; ShardKind kind; int socket;
  int row0, rows, col0, cols;         /* in the source tensor */
  void *q; float *s;                  /* destination values / Q8 scales in the socket region */
  size_t ld;                          /* destination row stride in elements (= cols) */
} Shard;
typedef struct { int sockets; Shard *t; int count; } ShardSet;

void shards_plan(ShardSet *set, const TensorManifest *m, int sockets, int gs);   /* validates */
void shards_load(ShardSet *set, int fd, Team *team);                             /* per-socket team */
```

The one-socket case is just `S = 1`: every tensor is `ROWS` or `COPY` with the full
range. A4's loader becomes `shards_load` with `S = 1`, so tracks A, B and C share a
single code path.

### 4.4 Packing

**`ROWS`.** One contiguous byte range per tensor per layer: values, then the
corresponding contiguous scale range. It is read with `pread` directly into the
destination.

**`COLS`.** Packed through bounded DDR scratch:
1. Read blocks of complete source rows (at most 64 MiB) into scratch in the owner
   socket's DDR.
2. Copy each row's `[col0, col0+cols)` slice into the destination row, with stride
   `cols`.

For Q8, copy the matching scale slice `[col0/gs, (col0+cols)/gs)` of each row too. The
copy is batched; there is never one `pread` per slice (study §5.4).

**Who loads each shard.** The socket team that owns the shard writes it, and every
destination region is `MPOL_BIND` to the owner's HBM node.

**Format record.** The report records the packing version, partition, source identity
(checkpoint SHA, or size plus mtime) and shard shapes.

## 5. Local tests

These run on macOS, on the `tp` fixture and on random tensors.

- **Reconstruction.** For S = 1 and S = 2, for FP32 and Q8 (gs 2, 4, 8), and for shared
  and unshared models, rebuild every full tensor from its shards. Every value and scale
  must be **bitwise** equal to the source.
- **Rejection.**
  - `gs = 3` with `dim/S = 8` is rejected.
  - `n_kv_heads = 1` with `S = 2` is rejected (the existing fixture).
- **Bounded scratch.** With a test-only 7-row block size, scratch never exceeds the
  limit.
- **Ownership.** With the virtual topology, each shard's destination belongs to the
  region of its socket.

## 6. Cluster run

`cresco8-tp.sbatch STAGE=check PHASE=B1`. **Estimated** at 45 minutes. It needs the
user's go-ahead to submit.

| Run | Gate |
| --- | --- |
| Shard 8B FP32 and INT8 with S = 2, teams pinned to sockets 0 and 1, HBM placement | Full reconstruction check (a test-only mode compares against the source file) |
| Verification | Socket 0 shards 100% on node 2; socket 1 shards 100% on node 3 |
| Startup breakdown | **Measured** row-copy and column-pack time per socket, and the scratch peak |

## 7. Exit criteria

- Reconstruction is bitwise exact for every tensor and scale.
- Per-socket placement is verified.
- Packing time and scratch peak are reported.
- Invalid partitions are rejected.

## 8. Risks and open questions

- **Column packing cost.** Wo and W2 hold about 1/3 of the layer bytes (**modeled**:
  `dim² + dim·hidden` out of the layer total). Packing them reads full rows through
  DDR. Expect the startup cost to grow by roughly the read time of those tensors.
  Measure it before considering a prepacked format.
- **Odd vocabulary split.** The classifier's vocabulary split with a remainder makes
  the logit gather buffer offsets non-uniform. Test it with an odd fixture vocabulary.

## 9. Not in this phase

- Computation on shards (B2–B5).
- AMX/BF16 packed layouts. C5 and C6 add pack hooks that consume these same `Shard`
  descriptors.
- Prepacked checkpoints.
