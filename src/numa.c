#include "numa.h"

#include "runtime.h"
#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#if defined(__has_include)
#if __has_include(<linux/mempolicy.h>)
#include <linux/mempolicy.h>
#endif
#endif
#endif
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

enum { MAX_CPUS = 1024, MAX_NODES = 128 };

#define FAIL(...)                                                                                                      \
  do {                                                                                                                 \
    snprintf(err, err_size, __VA_ARGS__);                                                                              \
    return -1;                                                                                                         \
  } while (0)

const char *numa_class_name(TopologyClass c) {
  switch (c) {
  case CLASS_NUMAPERF:
    return "numaperf";
  case CLASS_EXPLICIT:
    return "explicit";
  case CLASS_VIRTUAL:
    return "virtual";
  case CLASS_DDR_ONLY:
    return "ddr-only";
  }
  return "?";
}

/* ---- list parsing: "0-3,8,10-11" into a byte-per-id set -------------------------------------------------------- */

static int parse_list(const char *s, unsigned char *bits, int nbits) {
  memset(bits, 0, (size_t)nbits);
  for (;;) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')
      s++;
    if (!*s)
      return 0;
    char *end;
    long a = strtol(s, &end, 10), b;
    if (end == s || a < 0)
      return -1;
    b = a;
    s = end;
    if (*s == '-') {
      b = strtol(++s, &end, 10);
      if (end == s || b < a)
        return -1;
      s = end;
    }
    if (b >= nbits)
      return -1;
    for (long i = a; i <= b; i++)
      bits[i] = 1;
    while (*s == ' ' || *s == '\t')
      s++;
    if (*s == ',')
      s++;
    else if (*s && *s != '\n' && *s != '\r')
      return -1;
  }
}

static void format_list(const unsigned char *bits, int n, char *out, size_t size) {
  size_t used = 0;
  out[0] = 0;
  for (int i = 0; i < n; i++) {
    if (!bits[i])
      continue;
    int j = i;
    while (j + 1 < n && bits[j + 1])
      j++;
    int w = j > i ? snprintf(out + used, size - used, "%s%d-%d", used ? "," : "", i, j)
                  : snprintf(out + used, size - used, "%s%d", used ? "," : "", i);
    if (w < 0 || (size_t)w >= size - used) {
      out[size - 1] = 0;
      return;
    }
    used += (size_t)w;
    i = j;
  }
}

static int read_first_line(const char *path, char *buf, size_t size) {
  FILE *f = fopen(path, "r");
  if (!f)
    return -1;
  char *got = fgets(buf, (int)size, f);
  fclose(f);
  if (!got)
    buf[0] = 0;
  buf[strcspn(buf, "\r\n")] = 0;
  return 0;
}

static int path_f(char *out, size_t size, const char *root, const char *format, ...) {
  size_t n = (size_t)snprintf(out, size, "%s", root);
  va_list args;
  va_start(args, format);
  int w = vsnprintf(out + n, size - n, format, args);
  va_end(args);
  return w < 0 || (size_t)w >= size - n ? -1 : 0;
}

typedef struct {
  int online, package; /* package: of a CPU-bearing node, else -1 */
  int cpu_bearing;
  size_t mem_total;
} Node;

typedef struct {
  const char *root;
  unsigned char cpu_online[MAX_CPUS], cpu_allowed[MAX_CPUS], cpu_rep[MAX_CPUS];
  int cpu_pkg[MAX_CPUS];
  Node node[MAX_NODES];
  unsigned short dist[MAX_NODES][MAX_NODES];
  unsigned char possible[MAX_NODES], mems[MAX_NODES];
  int max_node;
  int pkgs[2], npkgs;
  int pkg_ddr[2], pkg_hbm[2]; /* indexed like pkgs */
} Machine;

static int pkg_index(const Machine *m, int package) {
  for (int i = 0; i < m->npkgs; i++)
    if (m->pkgs[i] == package)
      return i;
  return -1;
}

/* ---- sysfs enumeration ---------------------------------------------------------------------------------------- */

static int read_cpus(Machine *m, const char *allowed_cpus, char *err, size_t err_size) {
  char path[512], line[4096];
  if (path_f(path, sizeof(path), m->root, "/devices/system/cpu/online") || read_first_line(path, line, sizeof(line)))
    FAIL("cannot read %s", path);
  if (parse_list(line, m->cpu_online, MAX_CPUS))
    FAIL("unsupported online CPU list '%s' (at most %d CPUs)", line, MAX_CPUS);
  if (allowed_cpus) {
    if (parse_list(allowed_cpus, m->cpu_allowed, MAX_CPUS))
      FAIL("invalid allowed CPU list '%s'", allowed_cpus);
    for (int i = 0; i < MAX_CPUS; i++)
      m->cpu_allowed[i] &= m->cpu_online[i];
  } else
    memcpy(m->cpu_allowed, m->cpu_online, sizeof(m->cpu_allowed));
  for (int c = 0; c < MAX_CPUS; c++) {
    if (!m->cpu_online[c])
      continue;
    if (path_f(path, sizeof(path), m->root, "/devices/system/cpu/cpu%d/topology/physical_package_id", c) ||
        read_first_line(path, line, sizeof(line)))
      FAIL("cannot read %s", path);
    char *end;
    m->cpu_pkg[c] = (int)strtol(line, &end, 10);
    if (end == line || m->cpu_pkg[c] < 0)
      FAIL("%s: invalid physical package id '%s'", path, line);
    if (!m->cpu_allowed[c])
      continue;
    /* One CPU per physical core: the lowest allowed SMT sibling represents the core. */
    if (path_f(path, sizeof(path), m->root, "/devices/system/cpu/cpu%d/topology/thread_siblings_list", c) ||
        read_first_line(path, line, sizeof(line)))
      FAIL("cannot read %s", path);
    unsigned char siblings[MAX_CPUS];
    if (parse_list(line, siblings, MAX_CPUS))
      FAIL("%s: invalid sibling list '%s'", path, line);
    int lowest = c;
    for (int s = 0; s < c; s++)
      if (siblings[s] && m->cpu_allowed[s]) {
        lowest = s;
        break;
      }
    m->cpu_rep[c] = lowest == c;
  }
  return 0;
}

static int read_nodes(Machine *m, const char *allowed_mems, char *err, size_t err_size) {
  char path[512], line[4096];
  unsigned char online[MAX_NODES], cpus[MAX_CPUS];
  if (path_f(path, sizeof(path), m->root, "/devices/system/node/possible") || read_first_line(path, line, sizeof(line)))
    FAIL("cannot read %s", path);
  if (parse_list(line, m->possible, MAX_NODES))
    FAIL("unsupported possible node list '%s' (at most %d node ids)", line, MAX_NODES);
  m->max_node = -1;
  for (int n = 0; n < MAX_NODES; n++)
    if (m->possible[n])
      m->max_node = n;
  if (m->max_node < 0)
    FAIL("%s lists no possible nodes", path);
  if (path_f(path, sizeof(path), m->root, "/devices/system/node/online") || read_first_line(path, line, sizeof(line)))
    FAIL("cannot read %s", path);
  if (parse_list(line, online, MAX_NODES))
    FAIL("invalid online node list '%s'", line);
  int row[MAX_NODES], count = 0;
  for (int n = 0; n < MAX_NODES; n++)
    count += online[n];
  for (int n = 0; n < MAX_NODES; n++) {
    if (!online[n])
      continue;
    if (!m->possible[n])
      FAIL("node %d is online but not possible", n);
    Node *node = &m->node[n];
    node->online = 1;
    node->package = -1;
    if (path_f(path, sizeof(path), m->root, "/devices/system/node/node%d/cpulist", n) ||
        read_first_line(path, line, sizeof(line)) || parse_list(line, cpus, MAX_CPUS))
      FAIL("cannot read a valid CPU list from %s", path);
    for (int c = 0; c < MAX_CPUS; c++) {
      if (!cpus[c] || !m->cpu_online[c])
        continue;
      node->cpu_bearing = 1;
      if (node->package >= 0 && node->package != m->cpu_pkg[c])
        FAIL("node %d spans physical packages %d and %d", n, node->package, m->cpu_pkg[c]);
      node->package = m->cpu_pkg[c];
    }
    FILE *f;
    if (path_f(path, sizeof(path), m->root, "/devices/system/node/node%d/meminfo", n) || !(f = fopen(path, "r")))
      FAIL("cannot read %s", path);
    while (fgets(line, sizeof(line), f)) {
      char *at = strstr(line, "MemTotal:");
      if (at)
        node->mem_total = (size_t)strtoull(at + 9, NULL, 10) * 1024;
    }
    fclose(f);
    if (path_f(path, sizeof(path), m->root, "/devices/system/node/node%d/distance", n) ||
        read_first_line(path, line, sizeof(line)))
      FAIL("cannot read %s", path);
    int got = 0;
    for (char *s = line, *end; got < MAX_NODES; s = end, got++) {
      long v = strtol(s, &end, 10);
      if (end == s)
        break;
      row[got] = (int)v;
    }
    if (got != count)
      FAIL("%s lists %d distances for %d online nodes", path, got, count);
    int col = 0;
    for (int o = 0; o < MAX_NODES; o++)
      if (online[o])
        m->dist[n][o] = (unsigned short)row[col++];
  }
  if (allowed_mems) {
    if (parse_list(allowed_mems, m->mems, MAX_NODES))
      FAIL("invalid allowed memory node list '%s'", allowed_mems);
    for (int n = 0; n < MAX_NODES; n++)
      m->mems[n] &= m->possible[n];
  } else
    memcpy(m->mems, online, sizeof(m->mems));
  return 0;
}

/* Group nodes by package; refuse what this engine does not implement. */
static int read_packages(Machine *m, char *err, size_t err_size) {
  for (int c = 0; c < MAX_CPUS; c++) {
    if (!m->cpu_online[c] || pkg_index(m, m->cpu_pkg[c]) >= 0)
      continue;
    if (m->npkgs == 2)
      FAIL("more than two sockets (physical packages) are not supported");
    m->pkgs[m->npkgs++] = m->cpu_pkg[c];
  }
  if (m->npkgs == 2 && m->pkgs[0] > m->pkgs[1]) {
    int swap = m->pkgs[0];
    m->pkgs[0] = m->pkgs[1];
    m->pkgs[1] = swap;
  }
  if (!m->npkgs)
    FAIL("no online CPUs");
  for (int i = 0; i < 2; i++)
    m->pkg_ddr[i] = m->pkg_hbm[i] = -1;
  for (int n = 0; n < MAX_NODES; n++) {
    if (!m->node[n].online || !m->node[n].cpu_bearing)
      continue;
    int p = pkg_index(m, m->node[n].package);
    if (m->pkg_ddr[p] >= 0)
      FAIL("package %d has more than one CPU-bearing node (%d and %d): sub-NUMA clustering is not supported",
           m->pkgs[p], m->pkg_ddr[p], n);
    m->pkg_ddr[p] = n;
  }
  for (int p = 0; p < m->npkgs; p++)
    if (m->pkg_ddr[p] < 0)
      FAIL("package %d has no CPU-bearing NUMA node", m->pkgs[p]);
  return 0;
}

/* ---- classification ------------------------------------------------------------------------------------------- */

static int attribute(const Machine *m, int node, const char *name, long *value) {
  char path[512], line[64];
  if (path_f(path, sizeof(path), m->root, "/devices/system/node/node%d/access0/initiators/%s", node, name) ||
      read_first_line(path, line, sizeof(line)) || !line[0])
    return -1;
  char *end;
  *value = strtol(line, &end, 10);
  return end == line ? -1 : 0;
}

/* Initiator nodes of `node` (entries named node<k>). Returns their count, or -1 without the directory. */
static int initiators(const Machine *m, int node, int *out, int capacity) {
  char path[512];
  if (path_f(path, sizeof(path), m->root, "/devices/system/node/node%d/access0/initiators", node))
    return -1;
  DIR *d = opendir(path);
  if (!d)
    return -1;
  int count = 0, k;
  char tail;
  for (struct dirent *e; (e = readdir(d));)
    if (sscanf(e->d_name, "node%d%c", &k, &tail) == 1 && count < capacity)
      out[count++] = k;
  closedir(d);
  return count;
}

/* 0 = classified, 1 = numaperf data missing or incomplete, -1 = error. */
static int classify_numaperf(Machine *m, char *err, size_t err_size) {
  long bw[MAX_NODES] = {0};
  for (int n = 0; n < MAX_NODES; n++)
    if (m->node[n].online && (m->node[n].cpu_bearing || m->node[n].mem_total) &&
        attribute(m, n, "read_bandwidth", &bw[n]))
      return 1;
  for (int n = 0; n < MAX_NODES; n++) {
    if (!m->node[n].online || m->node[n].cpu_bearing || !m->node[n].mem_total)
      continue;
    int init[MAX_NODES], count = initiators(m, n, init, MAX_NODES), owner = -1;
    if (count <= 0)
      return 1;
    for (int i = 0; i < count; i++) {
      if (init[i] >= MAX_NODES || !m->node[init[i]].online || !m->node[init[i]].cpu_bearing)
        return 1;
      int p = pkg_index(m, m->node[init[i]].package);
      if (owner >= 0 && owner != p)
        FAIL("memory-only node %d has initiators in more than one socket: ambiguous", n);
      owner = p;
    }
    if (bw[n] <= bw[m->pkg_ddr[owner]])
      continue; /* memory-only but not faster than DDR: not HBM */
    if (m->pkg_hbm[owner] >= 0)
      FAIL("socket %d has more than one candidate HBM node (%d and %d): ambiguous", owner, m->pkg_hbm[owner], n);
    m->pkg_hbm[owner] = n;
  }
  return 0;
}

static int classify_explicit(Machine *m, const char *list, char *err, size_t err_size) {
  int c[2], h[2], d[2], count = 0, used;
  const char *s = list;
  while (*s) {
    if (count == 2)
      FAIL("explicit topology lists more than two sockets: '%s'", list);
    if (sscanf(s, "%d/%d/%d%n", &c[count], &h[count], &d[count], &used) != 3)
      FAIL("invalid explicit topology '%s': expected explicit:C0/H0/D0,C1/H1/D1 (package/hbm node/ddr node)", list);
    s += used;
    count++;
    if (*s == ',')
      s++;
    else if (*s)
      FAIL("invalid explicit topology '%s': expected a comma between sockets", list);
  }
  if (count != m->npkgs)
    FAIL("explicit topology describes %d socket(s) but the machine has %d package(s)", count, m->npkgs);
  for (int i = 0; i < count; i++) {
    int p = pkg_index(m, c[i]);
    if (p < 0)
      FAIL("explicit topology: package %d does not exist", c[i]);
    if (d[i] < 0 || d[i] >= MAX_NODES || !m->node[d[i]].online || !m->node[d[i]].cpu_bearing ||
        m->node[d[i]].package != c[i])
      FAIL("explicit topology: node %d is not the CPU-bearing DDR node of package %d", d[i], c[i]);
    if (h[i] < 0 || h[i] >= MAX_NODES || !m->node[h[i]].online || m->node[h[i]].cpu_bearing)
      FAIL("explicit topology: node %d is not an online CPU-less memory node", h[i]);
    m->pkg_ddr[p] = d[i];
    m->pkg_hbm[p] = h[i];
  }
  if (count == 2 && (m->pkg_hbm[0] == m->pkg_hbm[1] || m->pkg_ddr[0] == m->pkg_ddr[1]))
    FAIL("explicit topology: the two sockets share a node");
  /* Each socket's DDR must be nearer to its own HBM than to the other socket's. */
  for (int p = 0; p < count && count == 2; p++)
    if (m->dist[m->pkg_ddr[p]][m->pkg_hbm[p]] >= m->dist[m->pkg_ddr[p]][m->pkg_hbm[1 - p]])
      FAIL("explicit topology: node %d is not nearer to HBM node %d than to node %d", m->pkg_ddr[p], m->pkg_hbm[p],
           m->pkg_hbm[1 - p]);
  return 0;
}

/* ---- discovery ------------------------------------------------------------------------------------------------ */

static int discover_virtual(Topology *t, const char *shape, char *err, size_t err_size) {
  int sockets, threads, used = 0;
  if (sscanf(shape, "%dx%d%n", &sockets, &threads, &used) != 2 || shape[used] || sockets < 1 || sockets > 2 ||
      threads < 1)
    FAIL("invalid virtual topology 'virtual:%s': expected virtual:SxT with S in 1..2 and T >= 1", shape);
  t->classification = CLASS_VIRTUAL;
  t->max_node = -1;
  t->nsockets = sockets;
  for (int s = 0; s < sockets; s++)
    t->socket[s] = (SocketTopo){
        .package = -1, .ncpus = threads, .ddr_node = -1, .hbm_node = -1, .ddr_allowed = 1, .hbm_allowed = 1};
  return 0;
}

int numa_discover(Topology *t, const NumaInputs *in, char *err, size_t err_size) {
  memset(t, 0, sizeof(*t));
  t->max_node = -1;
  const char *spec = in->spec && *in->spec ? in->spec : "auto";
  if (!strncmp(spec, "virtual:", 8))
    return discover_virtual(t, spec + 8, err, err_size);
  int explicit_spec = !strncmp(spec, "explicit:", 9);
  if (!explicit_spec && strcmp(spec, "auto"))
    FAIL("invalid topology '%s': expected auto, virtual:SxT or explicit:C0/H0/D0,C1/H1/D1", spec);

  Machine *m = calloc(1, sizeof(*m));
  if (!m)
    FAIL("out of memory");
  m->root = in->sysfs_root && *in->sysfs_root ? in->sysfs_root : "/sys";
  int status = read_cpus(m, in->allowed_cpus, err, err_size) || read_nodes(m, in->allowed_mems, err, err_size) ||
               read_packages(m, err, err_size);
  int memory_only = 0;
  for (int n = 0; !status && n < MAX_NODES; n++)
    memory_only += m->node[n].online && !m->node[n].cpu_bearing && m->node[n].mem_total;
  if (!status) {
    if (explicit_spec) {
      status = classify_explicit(m, spec + 9, err, err_size);
      t->classification = CLASS_EXPLICIT;
    } else if (!memory_only) {
      t->classification = CLASS_DDR_ONLY;
    } else {
      int result = classify_numaperf(m, err, err_size);
      status = result;
      if (result == 1) {
        snprintf(err, err_size,
                 "cannot classify memory-only NUMA nodes: no numaperf bandwidth data in sysfs; set "
                 "LLAMA_TOPOLOGY=explicit:C0/H0/D0,C1/H1/D1 (package/HBM node/DDR node per socket, e.g. "
                 "explicit:0/2/0,1/3/1)");
        status = -1;
      }
      t->classification = CLASS_NUMAPERF;
    }
  }
  if (status) {
    free(m);
    return -1;
  }

  t->max_node = m->max_node;
  t->mems_words = ((size_t)m->max_node + 1 + 8 * sizeof(unsigned long) - 1) / (8 * sizeof(unsigned long));
  /* Plain allocations: the cached topology lives for the whole process and must not show up as a leak in the
   * engine's per-category accounting, which tests require to return to zero. */
  t->mems_allowed = calloc(t->mems_words, sizeof(unsigned long));
  if (!t->mems_allowed) {
    free(m);
    FAIL("out of memory");
  }
  for (int n = 0; n <= m->max_node; n++)
    if (m->mems[n])
      t->mems_allowed[n / (8 * sizeof(unsigned long))] |= 1ul << (n % (8 * sizeof(unsigned long)));
  format_list(m->mems, MAX_NODES, t->mems_allowed_list, sizeof(t->mems_allowed_list));

  for (int p = 0; p < m->npkgs; p++) {
    int count = 0;
    for (int c = 0; c < MAX_CPUS; c++)
      count += m->cpu_online[c] && m->cpu_rep[c] && m->cpu_pkg[c] == m->pkgs[p];
    if (!count)
      continue; /* this job has no CPU in this socket */
    SocketTopo *s = &t->socket[t->nsockets++];
    int d = m->pkg_ddr[p], h = m->pkg_hbm[p];
    *s = (SocketTopo){.package = m->pkgs[p],
                      .ncpus = count,
                      .cpus = calloc((size_t)count, sizeof(int)),
                      .ddr_node = d,
                      .hbm_node = h,
                      .ddr_allowed = d < 0 || m->mems[d],
                      .hbm_allowed = h < 0 || m->mems[h],
                      .ddr_bytes = d >= 0 ? m->node[d].mem_total : 0,
                      .hbm_bytes = h >= 0 ? m->node[h].mem_total : 0};
    for (int c = 0, i = 0; c < MAX_CPUS; c++)
      if (m->cpu_online[c] && m->cpu_rep[c] && m->cpu_pkg[c] == m->pkgs[p])
        s->cpus[i++] = c;
  }
  free(m);
  if (!t->nsockets) {
    numa_topology_free(t);
    FAIL("this job has no allowed CPU");
  }
  return 0;
}

void numa_topology_free(Topology *t) {
  for (int s = 0; s < 2; s++)
    free(t->socket[s].cpus);
  free(t->mems_allowed);
  memset(t, 0, sizeof(*t));
}

int numa_check_placement(const Topology *t, int sockets, const char *placement, char *err, size_t err_size) {
  if (!placement || !*placement)
    placement = "file";
  if (strcmp(placement, "file") && strcmp(placement, "ddr") && strcmp(placement, "hbm"))
    FAIL("invalid placement '%s': expected file, ddr or hbm", placement);
  if (sockets < 1 || sockets > 2)
    FAIL("invalid socket count %d: expected 1 or 2", sockets);
  if (t->nsockets < sockets)
    FAIL("%d socket(s) requested but this job has allowed CPUs in only %d", sockets, t->nsockets);
  if (!strcmp(placement, "file"))
    return 0;
  int hbm = !strcmp(placement, "hbm");
  for (int s = 0; s < sockets; s++) {
    const SocketTopo *k = &t->socket[s];
    int node = hbm ? k->hbm_node : k->ddr_node, allowed = hbm ? k->hbm_allowed : k->ddr_allowed;
    if (t->classification == CLASS_VIRTUAL)
      FAIL("placement %s needs real NUMA nodes, but the topology is virtual", placement);
    if (node < 0)
      FAIL("placement %s requested but socket %d (package %d) has no %s node (classification: %s)", placement, s,
           k->package, hbm ? "HBM" : "DDR", numa_class_name(t->classification));
    if (!allowed)
      FAIL("placement %s requested but node %d (socket %d %s) is not in this job's allowed memory nodes (%s)",
           placement, node, s, hbm ? "HBM" : "DDR", t->mems_allowed_list);
  }
  return 0;
}

/* ---- the job's real allowed sets and the process-wide topology ----------------------------------------------- */

int numa_affinity_cpus(int *cpus, int capacity) {
#ifdef __linux__
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof(set), &set))
    return -1;
  int n = 0;
  for (int i = 0; i < CPU_SETSIZE && n < capacity; i++)
    if (CPU_ISSET(i, &set))
      cpus[n++] = i;
  return n;
#else
  (void)cpus;
  (void)capacity;
  return -1;
#endif
}

int numa_online_nodes(char *buf, size_t size) {
#ifdef __linux__
  return read_first_line("/sys/devices/system/node/online", buf, size);
#else
  (void)buf;
  (void)size;
  return -1;
#endif
}

#ifdef __linux__
#ifndef MPOL_F_MEMS_ALLOWED
#define MPOL_F_MEMS_ALLOWED (1 << 2)
#endif
/* Allowed memory nodes of the calling thread as a list, from get_mempolicy with a mask sized for every
 * possible node (offline ids included); falls back to /proc/self/status. */
static int allowed_mems_list(char *out, size_t size) {
  enum { BITS = 1024 };
  unsigned long mask[BITS / (8 * sizeof(unsigned long))] = {0};
  if (syscall(SYS_get_mempolicy, NULL, mask, (unsigned long)BITS, NULL, (unsigned long)MPOL_F_MEMS_ALLOWED) == 0) {
    unsigned char bits[MAX_NODES];
    for (int n = 0; n < MAX_NODES; n++)
      bits[n] = (mask[n / (8 * sizeof(unsigned long))] >> (n % (8 * sizeof(unsigned long)))) & 1;
    format_list(bits, MAX_NODES, out, size);
    return 0;
  }
  return -1;
}

static void proc_mems_allowed(char *out, size_t size) {
  out[0] = 0;
  FILE *f = fopen("/proc/self/status", "r");
  if (!f)
    return;
  char line[512];
  while (fgets(line, sizeof(line), f))
    if (!strncmp(line, "Mems_allowed_list:", 18)) {
      snprintf(out, size, "%s", line + 18 + strspn(line + 18, " \t"));
      out[strcspn(out, "\r\n")] = 0;
    }
  fclose(f);
}
#endif

static Topology topology_cache;
static int topology_state; /* 0 untried, 1 discovered, -1 failed */
static char topology_error[512];

int numa_topology_try(const Topology **topology, char *err, size_t err_size) {
  if (topology_state == 0) {
    NumaInputs in = {.sysfs_root = getenv("LLAMA_SYSFS_ROOT"), .spec = getenv("LLAMA_TOPOLOGY")};
#ifdef __linux__
    /* Capture the full allowed sets before anything is pinned. A fixture root replaces them (tests only). */
    static char cpus[8192], mems[512];
    if (!in.sysfs_root) {
      cpu_set_t set;
      unsigned char bits[MAX_CPUS] = {0};
      if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE && i < MAX_CPUS; i++)
          bits[i] = CPU_ISSET(i, &set) != 0;
        format_list(bits, MAX_CPUS, cpus, sizeof(cpus));
        in.allowed_cpus = cpus;
      }
      if (allowed_mems_list(mems, sizeof(mems)) == 0)
        in.allowed_mems = mems;
    }
#endif
    if (numa_discover(&topology_cache, &in, topology_error, sizeof(topology_error)) == 0) {
#ifdef __linux__
      proc_mems_allowed(topology_cache.proc_mems_allowed_list, sizeof(topology_cache.proc_mems_allowed_list));
#endif
      topology_state = 1;
    } else
      topology_state = -1;
  }
  if (topology_state < 0) {
    snprintf(err, err_size, "%s", topology_error);
    return -1;
  }
  *topology = &topology_cache;
  return 0;
}

const Topology *numa_topology(void) {
  const Topology *topology;
  char err[512];
  if (numa_topology_try(&topology, err, sizeof(err)))
    fail("topology discovery failed: %s", err);
  return topology;
}

/* ---- bound regions (A2) ---------------------------------------------------------------------------------------- */

enum { REGION_LIMIT = 1024, THP_BYTES = 2 << 20 };
static NumaRegion live[REGION_LIMIT];
static int live_count, exit_hook;
static RegionRecord history[REGION_LIMIT];
static int history_count;

int region_history_count(void) {
  return history_count;
}
const RegionRecord *region_history_at(int index) {
  return index >= 0 && index < history_count ? &history[index] : NULL;
}
void region_record_verification(const NumaRegion *r, const PageCheck *check, size_t thp_bytes) {
  for (int i = history_count - 1; i >= 0; i--) /* the newest reservation at this address */
    if (history[i].id == r->mapping_base) {
      history[i].checked = 1;
      history[i].full = check->full;
      history[i].thp_bytes = thp_bytes;
      history[i].pages_queried = check->queried;
      history[i].pages_mismatched = check->wrong_node + check->errors;
      history[i].pages_missing = check->missing;
      return;
    }
}

const char *region_role_name(RegionRole role) {
  static const char *const names[] = {"weights", "packed", "embedding", "kv.k", "kv.v", "work", "scratch"};
  return (unsigned)role < sizeof(names) / sizeof(*names) ? names[role] : "?";
}

static MemoryKind region_kind(RegionRole role) {
  switch (role) {
  case ROLE_WEIGHTS:
    return MEM_WEIGHTS_OWNED;
  case ROLE_PACKED:
    return MEM_WEIGHTS_PACKED;
  case ROLE_EMBEDDING:
    return MEM_EMBEDDING;
  case ROLE_KV_K:
  case ROLE_KV_V:
    return MEM_KV;
  default:
    return MEM_ACTIVATION;
  }
}

int region_round(size_t bytes, size_t align, size_t *rounded) {
  if (!bytes || !align || bytes > SIZE_MAX - (align - 1))
    return -1;
  *rounded = (bytes + align - 1) / align * align;
  return 0;
}

int region_thp_default(void) {
  const char *thp = getenv("LLAMA_THP");
  return thp && !strcmp(thp, "on");
}

int region_live_count(void) {
  return live_count;
}
const NumaRegion *region_live_at(int index) {
  return index >= 0 && index < live_count ? &live[index] : NULL;
}

static NumaRegion *find_live(const void *mapping_base) {
  for (int i = 0; i < live_count; i++)
    if (live[i].mapping_base == mapping_base)
      return &live[i];
  return NULL;
}

static void drop_live(NumaRegion *entry) {
  if (entry->populated)
    memory_untrack(region_kind(entry->role), entry->logical_bytes);
  munmap(entry->mapping_base, entry->mapping_bytes);
  memmove(entry, entry + 1, (size_t)(&live[live_count] - (entry + 1)) * sizeof(*entry));
  live_count--;
}

void region_release(NumaRegion *r) {
  if (!r || !r->mapping_base)
    return;
  NumaRegion *entry = find_live(r->mapping_base);
  if (entry)
    drop_live(entry); /* an entry already released by region_release_all is not unmapped twice */
  memset(r, 0, sizeof(*r));
  r->node = -1;
}

void region_release_all(void) {
  while (live_count)
    drop_live(&live[live_count - 1]);
}

#ifdef __linux__
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif

static int possible_max_node(void) {
  static int cached = -2;
  if (cached == -2) {
    char line[512];
    unsigned char possible[MAX_NODES];
    cached = -1;
    if (read_first_line("/sys/devices/system/node/possible", line, sizeof(line)) == 0 &&
        parse_list(line, possible, MAX_NODES) == 0)
      for (int n = 0; n < MAX_NODES; n++)
        if (possible[n])
          cached = n;
  }
  return cached;
}

static size_t node_capacity(int node) {
  char path[128], line[256];
  snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
  FILE *f = fopen(path, "r");
  size_t total = 0;
  while (f && fgets(line, sizeof(line), f)) {
    char *at = strstr(line, "MemTotal:");
    if (at)
      total = (size_t)strtoull(at + 9, NULL, 10) * 1024;
  }
  if (f)
    fclose(f);
  return total;
}

/* Test-only fault injection (LLAMA_NUMA_FAULT=EPERM|ENOSYS|EINVAL), checked where the syscall would be made. */
static int injected_fault(void) {
  const char *fault = getenv("LLAMA_NUMA_FAULT");
  if (!fault || !*fault)
    return 0;
  return !strcmp(fault, "EPERM") ? EPERM : !strcmp(fault, "ENOSYS") ? ENOSYS : !strcmp(fault, "EINVAL") ? EINVAL : EIO;
}

static int bind_memory(void *addr, size_t bytes, int node, int max_node) {
  int fault = injected_fault();
  if (fault)
    return fault;
  unsigned long mask[MAX_NODES / (8 * sizeof(unsigned long)) + 1] = {0};
  mask[node / (8 * sizeof(unsigned long))] = 1ul << (node % (8 * sizeof(unsigned long)));
  /* The kernel reads maxnode - 1 bits of the mask, so a node id of max_node needs maxnode = max_node + 2, as libnuma
   * passes it. A plain bit count (max_node + 1) would silently drop the highest node. */
  long rc = syscall(SYS_mbind, addr, bytes, MPOL_BIND, mask, (unsigned long)max_node + 2, 0u);
  return rc == 0 ? 0 : errno;
}
#endif

int region_reserve_try(NumaRegion *r, size_t bytes, int socket, int node, RegionRole role, int thp, char *err,
                       size_t err_size) {
  memset(r, 0, sizeof(*r));
  r->node = -1;
  int bound = node >= 0;
#ifndef __linux__
  if (bound)
    FAIL("cannot bind memory to NUMA node %d: explicit placement requires Linux", node);
#endif
  size_t page = (size_t)sysconf(_SC_PAGESIZE), align = bound && thp ? (size_t)THP_BYTES : page, rounded;
  if (region_round(bytes, align, &rounded))
    FAIL("invalid region size %zu bytes (zero, or overflows when rounded to %zu)", bytes, align);
  size_t slack = align > page ? align : 0, mapping;
  if (rounded > SIZE_MAX - slack || rounded + slack > PTRDIFF_MAX)
    FAIL("region of %zu bytes exceeds the addressable size", bytes);
  mapping = rounded + slack;
#ifdef __linux__
  int max_node = -1;
  if (bound) {
    max_node = possible_max_node();
    if (max_node < 0 || node > max_node)
      FAIL("NUMA node %d is not a possible node (highest possible: %d)", node, max_node);
    size_t capacity = node_capacity(node), reserved = 0;
    for (int i = 0; i < live_count; i++)
      reserved += live[i].node == node ? live[i].data_bytes : 0;
    if (capacity && reserved + rounded > capacity)
      FAIL("region of %zu bytes would exceed node %d capacity (%zu bytes total, %zu already reserved)", rounded, node,
           capacity, reserved);
  }
#endif
  if (live_count == REGION_LIMIT)
    FAIL("too many live regions (limit %d)", REGION_LIMIT);
  /* No MAP_POPULATE and no write: the pages must not exist before the policy does. */
  void *base = mmap(NULL, mapping, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base == MAP_FAILED)
    FAIL("cannot reserve %zu bytes: %s", mapping, strerror(errno));
  uintptr_t start = ((uintptr_t)base + align - 1) / align * align;
  void *data = (void *)start;
#ifdef __linux__
  if (bound) {
    int status = bind_memory(data, rounded, node, max_node);
    if (status) {
      char allowed[512] = "unknown";
      allowed_mems_list(allowed, sizeof(allowed));
      munmap(base, mapping);
      FAIL("cannot bind %zu bytes to NUMA node %d: %s (errno %d); allowed memory nodes: %s", rounded, node,
           strerror(status), status, allowed);
    }
#ifdef MADV_HUGEPAGE
    if (thp && madvise(data, rounded, MADV_HUGEPAGE) == 0)
      r->thp_requested = 1;
    else if (thp)
      LOG(LOG_WARN, "madvise(MADV_HUGEPAGE) failed for a region on node %d: %s", node, strerror(errno));
#endif
  }
#endif
  *r = (NumaRegion){.mapping_base = base,
                    .mapping_bytes = mapping,
                    .data = data,
                    .logical_bytes = bytes,
                    .data_bytes = rounded,
                    .socket = socket,
                    .node = node,
                    .role = role,
                    .backend = bound ? BACKEND_LINUX_BOUND : BACKEND_PORTABLE,
                    .thp_requested = r->thp_requested};
  live[live_count++] = *r;
  if (history_count < REGION_LIMIT)
    history[history_count++] = (RegionRecord){
        .id = base, .role = role, .socket = socket, .node = node, .thp_requested = r->thp_requested, .bytes = bytes};
  if (!exit_hook) {
    exit_hook = 1;
    atexit(region_release_all);
  }
  return 0;
}

void region_reserve(NumaRegion *r, size_t bytes, int socket, int node, RegionRole role, int thp) {
  char err[512];
  if (region_reserve_try(r, bytes, socket, node, role, thp, err, sizeof(err)))
    fail("%s", err);
}

void region_mark_populated(NumaRegion *r) {
  NumaRegion *entry = r->mapping_base ? find_live(r->mapping_base) : NULL;
  if (!entry || entry->populated)
    return;
  entry->populated = r->populated = 1;
  memory_track(region_kind(r->role), r->logical_bytes);
}

void region_protect(NumaRegion *r) {
  NumaRegion *entry = r->mapping_base ? find_live(r->mapping_base) : NULL;
  if (!entry)
    fail("cannot protect a region that is not live");
  if (mprotect(r->data, r->data_bytes, PROT_READ))
    fail("cannot make region read-only: %s", strerror(errno));
  entry->read_only = r->read_only = 1;
}

/* ---- placement verification (A3) -------------------------------------------------------------------------------- */

void page_check_classify(const int *status, size_t n, int node, PageCheck *out) {
  for (size_t i = 0; i < n; i++) {
    out->queried++;
    if (status[i] >= 0)
      status[i] == node ? out->on_node++ : out->wrong_node++;
    else if (status[i] == -ENOENT || status[i] == -EFAULT)
      out->missing++; /* not populated: untouched anonymous pages report -EFAULT on current kernels */
    else
      out->errors++;
  }
}

/* Endpoint j of the extent list: even j is the first page of extent j/2, odd j its last. */
static size_t extent_endpoint(size_t j, size_t pages, size_t extent_pages) {
  size_t extent = j / 2, first = extent * extent_pages,
         end = first + extent_pages < pages ? first + extent_pages : pages;
  return j % 2 ? end - 1 : first;
}

static int compare_size(const void *a, const void *b) {
  size_t x = *(const size_t *)a, y = *(const size_t *)b;
  return x < y ? -1 : x > y;
}

size_t page_sample_plan(size_t pages, size_t extent_pages, size_t max_samples, size_t *out) {
  if (!pages || !max_samples)
    return 0;
  if (!extent_pages)
    extent_pages = 1;
  size_t endpoints = 2 * ((pages + extent_pages - 1) / extent_pages), count = 0;
  size_t picks = endpoints < max_samples ? endpoints : max_samples;
  for (size_t i = 0; i < picks; i++) /* evenly spaced endpoints, always including the first and the last page */
    out[count++] = extent_endpoint(picks > 1 ? i * (endpoints - 1) / (picks - 1) : 0, pages, extent_pages);
  size_t fill = max_samples - picks;
  for (size_t i = 0; i < fill; i++) /* deterministic stride over the whole region */
    out[count++] = (size_t)((unsigned long long)i * pages / fill);
  qsort(out, count, sizeof(*out), compare_size);
  size_t unique = 0;
  for (size_t i = 0; i < count; i++)
    if (!unique || out[i] != out[unique - 1])
      out[unique++] = out[i];
  return unique;
}

static int hex_range(const char *line, uintptr_t *lo, uintptr_t *hi) {
  char *end;
  unsigned long long a = strtoull(line, &end, 16);
  if (end == line || *end != '-')
    return -1;
  unsigned long long b = strtoull(end + 1, &end, 16);
  if (*end != ' ')
    return -1;
  *lo = (uintptr_t)a;
  *hi = (uintptr_t)b;
  return 0;
}

/* AnonHugePages is per VMA and VMAs merge, so a VMA that only partly overlaps the region contributes
 * proportionally: an estimate there, exact where the region covers whole VMAs. */
size_t smaps_thp_bytes(const char *text, uintptr_t lo, uintptr_t hi) {
  double total = 0;
  uintptr_t start = 0, end = 0;
  int inside = 0;
  for (const char *line = text; *line;) {
    uintptr_t a, b;
    if (hex_range(line, &a, &b) == 0) {
      start = a;
      end = b;
      inside = b > a;
    } else if (inside && !strncmp(line, "AnonHugePages:", 14)) {
      double bytes = (double)strtoull(line + 14, NULL, 10) * 1024;
      uintptr_t from = start > lo ? start : lo, to = end < hi ? end : hi;
      if (to > from)
        total += bytes * (double)(to - from) / (double)(end - start);
    }
    const char *next = strchr(line, '\n');
    if (!next)
      break;
    line = next + 1;
  }
  return (size_t)(total + 0.5);
}

/* numa_maps lines start at the VMA's first address: the VMA holding `addr` is the last line starting at or below it. */
int numa_maps_lookup(const char *text, uintptr_t addr, char *policy, size_t policy_size, size_t *node_pages,
                     int nodes) {
  const char *best = NULL;
  for (const char *line = text; *line;) {
    char *end;
    unsigned long long start = strtoull(line, &end, 16);
    if (end != line && *end == ' ' && (uintptr_t)start <= addr)
      best = line;
    const char *next = strchr(line, '\n');
    if (!next)
      break;
    line = next + 1;
  }
  if (!best)
    return -1;
  for (int i = 0; i < nodes; i++)
    node_pages[i] = 0;
  size_t length = strcspn(best, "\n");
  char copy[1024];
  snprintf(copy, sizeof(copy), "%.*s", (int)(length < sizeof(copy) - 1 ? length : sizeof(copy) - 1), best);
  char *save, *token = strtok_r(copy, " ", &save);
  token = strtok_r(NULL, " ", &save);
  snprintf(policy, policy_size, "%s", token ? token : "");
  while ((token = strtok_r(NULL, " ", &save)))
    if (token[0] == 'N' && token[1] >= '0' && token[1] <= '9') {
      char *eq;
      long node = strtol(token + 1, &eq, 10);
      if (*eq == '=' && node < nodes)
        node_pages[node] = (size_t)strtoull(eq + 1, NULL, 10);
    }
  return 0;
}

size_t region_verify_samples_default(void) {
  const char *value = getenv("LLAMA_VERIFY_SAMPLES");
  return value ? (size_t)strtoull(value, NULL, 10) : 0;
}

int placement_strict(const char *placement) {
  const char *value = getenv("LLAMA_PLACEMENT_STRICT");
  if (value && *value)
    return strcmp(value, "0") != 0;
  return placement && (!strcmp(placement, "hbm") || !strcmp(placement, "ddr"));
}

#ifdef __linux__
static char *read_whole_file(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f)
    return NULL;
  size_t size = 0, capacity = 1 << 16;
  char *text = malloc(capacity);
  while (text) {
    size += fread(text + size, 1, capacity - size - 1, f);
    if (size < capacity - 1)
      break;
    char *grown = realloc(text, capacity *= 2);
    if (!grown)
      free(text);
    text = grown;
  }
  fclose(f);
  if (text)
    text[size] = 0;
  return text;
}
#endif

int region_thp_bytes(const NumaRegion *r, size_t *bytes) {
#ifdef __linux__
  char *text = read_whole_file("/proc/self/smaps");
  if (!text)
    return -1;
  *bytes = smaps_thp_bytes(text, (uintptr_t)r->data, (uintptr_t)r->data + r->data_bytes);
  free(text);
  return 0;
#else
  (void)r;
  (void)bytes;
  return -1;
#endif
}

int region_numa_maps(const NumaRegion *r, char *policy, size_t policy_size, size_t *node_pages, int nodes) {
#ifdef __linux__
  char *text = read_whole_file("/proc/self/numa_maps");
  if (!text)
    return -1;
  int status = numa_maps_lookup(text, (uintptr_t)r->data, policy, policy_size, node_pages, nodes);
  free(text);
  return status;
#else
  (void)r;
  (void)policy;
  (void)policy_size;
  (void)node_pages;
  (void)nodes;
  return -1;
#endif
}

int region_verify(const NumaRegion *r, size_t max_samples, PageCheck *out) {
  *out = (PageCheck){.scheme = "full"};
#ifdef __linux__
  if (!r->mapping_base || r->node < 0)
    return -1;
  enum { BATCH = 4096 };
  size_t page = (size_t)sysconf(_SC_PAGESIZE), pages = (r->logical_bytes + page - 1) / page;
  void *addresses[BATCH];
  int status[BATCH];
  size_t *plan = NULL, plan_count = 0;
  out->full = max_samples == 0 || pages <= max_samples;
  if (!out->full) {
    out->scheme = "stride+extent-endpoints";
    plan = memory_alloc(max_samples, sizeof(*plan), MEM_METADATA);
    plan_count = page_sample_plan(pages, (size_t)(2 << 20) / page, max_samples, plan);
  }
  size_t total = out->full ? pages : plan_count;
  for (size_t done = 0; done < total;) {
    size_t n = total - done < BATCH ? total - done : BATCH;
    for (size_t i = 0; i < n; i++)
      addresses[i] = (char *)r->data + (out->full ? done + i : plan[done + i]) * page;
    long rc = syscall(SYS_move_pages, 0, (unsigned long)n, addresses, NULL, status, 0);
    if (rc != 0) {
      memory_free(plan);
      *out = (PageCheck){.scheme = "unavailable", .errors = n};
      return -1;
    }
    page_check_classify(status, n, r->node, out);
    done += n;
  }
  memory_free(plan);
  return 0;
#else
  (void)r;
  (void)max_samples;
  return -1;
#endif
}
