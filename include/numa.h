#ifndef LLAMA_NUMA_H
#define LLAMA_NUMA_H

#include <stddef.h>

/* Socket/NUMA topology discovered in C from sysfs and the job's allowed sets, with
 * no numactl or lscpu. A node is never classified as HBM merely because it is
 * memory-only: that needs numaperf data or an explicit map. */
typedef enum {
  CLASS_NUMAPERF, /* HBM nodes identified from /sys/.../access0/initiators bandwidth */
  CLASS_EXPLICIT, /* LLAMA_TOPOLOGY=explicit:C0/H0/D0,C1/H1/D1 */
  CLASS_VIRTUAL,  /* LLAMA_TOPOLOGY=virtual:SxT: fake unpinned sockets, no nodes */
  CLASS_DDR_ONLY  /* no CPU-less memory node exists, so there is nothing to classify */
} TopologyClass;

typedef struct {
  int package; /* physical_package_id; -1 for virtual sockets */
  int ncpus;
  int *cpus;                    /* allowed physical cores, one CPU per core, ascending; NULL when virtual */
  int ddr_node, hbm_node;       /* -1 when absent */
  int ddr_allowed, hbm_allowed; /* node is in this job's allowed memory set (1 when virtual or absent) */
  size_t ddr_bytes, hbm_bytes;  /* node MemTotal, informational */
} SocketTopo;

typedef struct {
  int nsockets; /* sockets that own at least one allowed CPU */
  SocketTopo socket[2];
  int max_node; /* highest possible node id, for mbind masks; -1 when virtual */
  size_t mems_words;
  unsigned long *mems_allowed;      /* bitmask of allowed nodes, mems_words words sized from the possible nodes */
  char mems_allowed_list[128];      /* the same set as a list, e.g. "0-3" */
  char proc_mems_allowed_list[128]; /* /proc/self/status Mems_allowed_list, diagnostic only (numa_topology) */
  TopologyClass classification;
} Topology;

/* Inputs of one discovery. NULL fields mean: real /sys, `auto`, every online CPU,
 * every online node. Tests substitute fixtures here. */
typedef struct {
  const char *sysfs_root;   /* replaces "/sys" */
  const char *spec;         /* LLAMA_TOPOLOGY: auto | virtual:SxT | explicit:C0/H0/D0,C1/H1/D1 (C = package id) */
  const char *allowed_cpus; /* cpu list, e.g. "0-55" */
  const char *allowed_mems; /* node list, e.g. "0,2" */
} NumaInputs;

/* Returns 0, or -1 with a message in `err`. */
int numa_discover(Topology *t, const NumaInputs *in, char *err, size_t err_size);
void numa_topology_free(Topology *t);
/* Whether `sockets` (1 or 2) sockets can run with `placement` (file|ddr|hbm; NULL = file). */
int numa_check_placement(const Topology *t, int sockets, const char *placement, char *err, size_t err_size);
const char *numa_class_name(TopologyClass c);

/* Discovered once, before any pinning, from LLAMA_TOPOLOGY and the job's real affinity
 * and allowed memory nodes; fails the process on error. Call before threads are created. */
const Topology *numa_topology(void);

/* Linux affinity and NUMA online read shared with the report; -1 when unavailable. */
int numa_affinity_cpus(int *cpus, int capacity);
int numa_online_nodes(char *buf, size_t size);

#endif
