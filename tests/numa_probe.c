/* Dump the discovered topology as JSON; used by tests (sysfs fixtures) and by the cluster check (real machine).
 *   numa_probe                          real /sys, real affinity and allowed nodes, LLAMA_TOPOLOGY
 *   numa_probe --sysfs ROOT [--topology SPEC] [--cpus LIST] [--mems LIST]
 *   either form: [--sockets N] [--placement file|ddr|hbm] checks that the layout can run */
#include "numa.h"
#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_list(const int *values, int n) {
  putchar('[');
  for (int i = 0; i < n; i++)
    printf("%s%d", i ? ", " : "", values[i]);
  putchar(']');
}

static void print_topology(const Topology *t) {
  printf("{\n  \"classification\": \"%s\",\n  \"nsockets\": %d,\n  \"max_node\": %d,\n",
         numa_class_name(t->classification), t->nsockets, t->max_node);
  printf("  \"mems_allowed_list\": \"%s\",\n  \"proc_mems_allowed_list\": \"%s\",\n  \"mems_allowed\": [",
         t->mems_allowed_list, t->proc_mems_allowed_list);
  int first = 1;
  for (int n = 0; n <= t->max_node; n++)
    if (t->mems_allowed[n / (8 * sizeof(unsigned long))] >> (n % (8 * sizeof(unsigned long))) & 1) {
      printf("%s%d", first ? "" : ", ", n);
      first = 0;
    }
  printf("],\n  \"sockets\": [\n");
  for (int s = 0; s < t->nsockets; s++) {
    const SocketTopo *k = &t->socket[s];
    printf("    {\"package\": %d, \"ncpus\": %d, \"cpus\": ", k->package, k->ncpus);
    if (k->cpus)
      print_list(k->cpus, k->ncpus);
    else
      printf("null");
    printf(", \"ddr_node\": %d, \"hbm_node\": %d, \"ddr_allowed\": %s, \"hbm_allowed\": %s, \"ddr_bytes\": %zu, "
           "\"hbm_bytes\": %zu}%s\n",
           k->ddr_node, k->hbm_node, k->ddr_allowed ? "true" : "false", k->hbm_allowed ? "true" : "false", k->ddr_bytes,
           k->hbm_bytes, s + 1 < t->nsockets ? "," : "");
  }
  printf("  ]\n}\n");
}

int main(int argc, char **argv) {
  NumaInputs in = {0};
  const char *placement = NULL;
  int sockets = 0;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      fprintf(stderr, "usage: numa_probe [--sysfs ROOT] [--topology SPEC] [--cpus LIST] [--mems LIST] [--sockets N] "
                      "[--placement P]\n");
      return 2;
    }
    if (!strcmp(argv[i], "--sysfs"))
      in.sysfs_root = argv[i + 1];
    else if (!strcmp(argv[i], "--topology"))
      in.spec = argv[i + 1];
    else if (!strcmp(argv[i], "--cpus"))
      in.allowed_cpus = argv[i + 1];
    else if (!strcmp(argv[i], "--mems"))
      in.allowed_mems = argv[i + 1];
    else if (!strcmp(argv[i], "--sockets"))
      sockets = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--placement"))
      placement = argv[i + 1];
    else {
      fprintf(stderr, "unknown argument '%s'\n", argv[i]);
      return 2;
    }
  }
  char err[512];
  Topology local;
  const Topology *t;
  int own = 0;
  if (in.sysfs_root || in.allowed_cpus || in.allowed_mems) {
    if (numa_discover(&local, &in, err, sizeof(err))) {
      fprintf(stderr, "[ERROR] %s\n", err);
      return 1;
    }
    t = &local;
    own = 1;
  } else {
    if (in.spec)
      setenv("LLAMA_TOPOLOGY", in.spec, 1);
    t = numa_topology();
  }
  print_topology(t);
  int status = 0;
  if ((sockets || placement) && numa_check_placement(t, sockets ? sockets : 1, placement, err, sizeof(err))) {
    fprintf(stderr, "[ERROR] %s\n", err);
    status = 1;
  }
  if (own)
    numa_topology_free(&local);
  return status;
}
