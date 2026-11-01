# A1. Topology discovery

## 1. Goal and status

Discover in C, without invoking `numactl` or `lscpu`, which CPUs, DDR node and HBM node
belong to each socket. Restrict the result to what this job is allowed to use. Refuse
any layout that cannot be classified safely. Classification is never inferred from a
node merely being memory-only (study §4).

Status: **planned**. Branch: `feature/numa-a1-topology`.

## 2. Depends on / unblocks

- Depends on: nothing. If F2 is in place, its pool consumes A1's CPU lists; until then
  A1 runs standalone.
- Unblocks: A2, the F2 pinning defaults, and the report's `topology` object (F3).

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| Linux affinity and NUMA online read | `src/report.c:156-186` | Move into `numa.c` and have the report call it |
| `fail`, `log_message` | `src/runtime.c:30-55` | Error reporting |
| hbm14 layout | `scripts/cresco8-kernels.sbatch:54-67`, hw-bench `docs/01_SCOPE_AND_TOPOLOGY.md` | Expected values: DDR nodes 0/1 with CPUs, CPU-less HBM nodes 2/3, distance 13 local and 23 remote |

## 4. Changes

### 4.1 API

`include/numa.h` and `src/numa.c`:

```c
/* Proposed. */
typedef enum { CLASS_NUMAPERF, CLASS_EXPLICIT, CLASS_VIRTUAL } TopologyClass;
typedef struct {
  int ncpus; int *cpus;                 /* allowed physical cores, one per core */
  int ddr_node, hbm_node;               /* -1 when absent */
  size_t ddr_bytes, hbm_bytes;          /* node MemTotal, informational */
} SocketTopo;
typedef struct {
  int nsockets; SocketTopo socket[2];
  int max_node;                         /* highest possible node id, for mbind masks */
  unsigned long *mems_allowed;          /* bitmask sized from possible nodes */
  TopologyClass classification;
} Topology;

const Topology *numa_topology(void);    /* discovered once, before any pinning */
```

### 4.2 Discovery steps (study §4, steps 1–6)

1. **Capture the allowed CPUs first.** Call `sched_getaffinity` on the launching
   thread before anything is pinned. This is the job's full allowed set.
2. **Enumerate CPUs.** Read
   `/sys/devices/system/cpu/cpu*/topology/{physical_package_id,core_id,thread_siblings_list}`
   for online CPUs. Keep one CPU per physical core. SMT is off on CRESCO8, but the
   code must not assume that.
3. **Enumerate memory nodes.** Read `/sys/devices/system/node/{possible,online}`, plus
   `node*/cpulist`, `node*/meminfo` (MemTotal) and `node*/distance`.
4. **Read the allowed memory nodes.** Use
   `syscall(SYS_get_mempolicy, NULL, mask, maxnode, NULL, MPOL_F_MEMS_ALLOWED)`, with
   the mask sized from `possible` and including offline-ID holes. Record
   `/proc/self/status` `Mems_allowed_list` as a diagnostic.
5. **Classify the nodes.**
   - **numaperf.** Where `/sys/devices/system/node/node*/access0/initiators/*` and the
     `read_bandwidth`/`read_latency` attributes exist, a CPU-less node whose best
     initiator is socket s, and whose bandwidth is above that socket's CPU-bearing
     node, is s's HBM.
   - **explicit.** Otherwise `LLAMA_TOPOLOGY=explicit:C0/H0/D0,C1/H1/D1` is required,
     for example `explicit:0/2/0,1/3/1`. It is checked against the discovered nodes:
     CPU-bearing DDR, CPU-less HBM candidate, and the distance relation.
   - **auto with no numaperf data and no explicit map** fails with a message that
     names the explicit form. A node is never classified silently.
6. **Reject ambiguity.** Fail when there is more than one CPU-bearing node per package
   (SNC), more than one candidate HBM node per socket, or more than two sockets. These
   need their own implementation and are out of scope for every phase.

### 4.3 Integration

- **Virtual topology.** `LLAMA_TOPOLOGY=virtual:SxT` short-circuits to S fake sockets
  with T unpinned threads each, no nodes, and `CLASS_VIRTUAL`. It works on any OS.
- **Thread defaults.** F2's pool uses `socket[s].cpus` as its default pinning lists.
- **Report.** The report writes the `topology` object.

## 5. Local tests

These run on macOS. The sysfs root is overridable through
`LLAMA_SYSFS_ROOT` (tests only).

- `tests/fixtures/sysfs-flat/`, a minimal tree copied in shape from hbm14 with 4 nodes
  and 112 CPUs (contents synthetic). Expected result: S0 = cores 0–55, DDR 0, HBM 2;
  S1 = cores 56–111, DDR 1, HBM 3.
- `sysfs-cache/`: 2 nodes, no HBM. With `LLAMA_PLACEMENT=hbm` this must fail with a
  clear message.
- `sysfs-snc4/`: 8 CPU-bearing nodes. Must be rejected.
- `sysfs-no-numaperf/`: `auto` fails, and `explicit:0/2/0,1/3/1` succeeds.
- Offline node-ID hole (`possible` 0–7, `online` 0–3): mask sizing is correct.
- A restricted `mems_allowed` that excludes node 3 is reported as such, and a
  two-socket request then fails.

## 6. Cluster run

`cresco8-placement.sbatch STAGE=check PHASE=A1`. **Estimated** at 10 minutes. It needs
the user's go-ahead to submit.

1. Dump `numa_topology()` as JSON (a small `tests/numa_probe.c` mode).
2. Compare it with `numactl -H` and the `flat-mode.json` produced by the existing check.
3. Run it again under `numactl --cpunodebind=0`, `srun --cpu-bind=cores` and a
   restricted `--membind=0,2`, to confirm the allowed-set handling.

Outputs are `topology-*.json` files and a diff summary.

## 7. Exit criteria

- On hbm14 the classification is correct: through numaperf, or through the explicit
  map if numaperf attributes are absent. Record which.
- Every allowed-set variation is reflected correctly.
- All fixture tests pass on macOS and Linux.

## 8. Risks and open questions

- **Missing numaperf data.** CRESCO8 firmware may not populate the numaperf sysfs
  attributes. Find out in the first `check` run. If they are absent, the explicit map
  becomes the documented way to run on CRESCO8, set by the sbatch script and recorded
  in the report.
- **Slurm cgroups.** The cgroup may restrict `mems` differently from `cpus`. Read both;
  do not assume them from the CPU allocation.

## 9. Not in this phase

- Binding memory (A2).
- Pinning threads (F2 consumes this phase's output).
- SNC layouts.
