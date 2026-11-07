#include "workers.h"

#include "runtime.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <string.h>

/* Spins before a waiter starts yielding. Futex parking is deferred until measurements justify it. */
enum { SPIN_LIMIT = 2048 };

/* Hot counters sit 64 bytes apart so waiters on one do not slow writers of another. */
typedef struct {
  atomic_uint value;
  char pad[60];
} Counter;

typedef struct {
  Team *team;
  int worker;
} WorkerArg;

struct Team {
  int index, size;
  pthread_t *threads; /* size - 1 workers; the leader is worker 0 */
  WorkerArg *args;
  const int *cpus;
  int *assigned, *observed; /* per worker; observed is written by that worker before it reports ready */
  TeamFn fn;
  const void *job;
  Counter generation; /* leader publishes a job by bumping it */
  Counter done;       /* workers finished with the current job */
  Counter ready;      /* workers that finished starting up */
  Counter stop;
  Counter barrier_count, barrier_generation;
  unsigned published; /* leader's copy of generation */
};

static Team *teams[2];
static int team_count;
static TeamSummary summaries[2];
static int summary_count;
static _Thread_local int in_pool_thread;
static atomic_int abort_flag;

void workers_abort(void) {
  atomic_store_explicit(&abort_flag, 1, memory_order_release);
}
int workers_aborted(void) {
  return atomic_load_explicit(&abort_flag, memory_order_acquire);
}
void workers_abort_clear(void) {
  atomic_store_explicit(&abort_flag, 0, memory_order_release);
}

static inline void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

/* Waits until *counter != seen and returns the new value. */
static unsigned wait_change(Counter *counter, unsigned seen) {
  for (unsigned spins = 0;; spins++) {
    unsigned now = atomic_load_explicit(&counter->value, memory_order_acquire);
    if (now != seen)
      return now;
    if (spins < SPIN_LIMIT)
      relax();
    else
      sched_yield();
  }
}

static void wait_reach(Counter *counter, unsigned target) {
  for (unsigned spins = 0; atomic_load_explicit(&counter->value, memory_order_acquire) != target; spins++) {
    if (spins < SPIN_LIMIT)
      relax();
    else
      sched_yield();
  }
}

/* Pins the calling thread to `cpu` and returns the CPU it reads back as, or -1. */
static int pin_self(int cpu) {
#ifdef __linux__
  if (cpu < 0 || cpu >= CPU_SETSIZE)
    return -1;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set))
    return -1;
  CPU_ZERO(&set);
  if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set))
    return -1;
  int found = -1, bits = 0;
  for (int i = 0; i < CPU_SETSIZE; i++)
    if (CPU_ISSET(i, &set)) {
      found = i;
      bits++;
    }
  return bits == 1 ? found : -1;
#else
  (void)cpu;
  return -1;
#endif
}

static void *worker_main(void *p) {
  Team *team = ((WorkerArg *)p)->team;
  int worker = ((WorkerArg *)p)->worker;
  in_pool_thread = 1;
  if (team->cpus)
    team->observed[worker] = pin_self(team->cpus[worker]);
  atomic_fetch_add_explicit(&team->ready.value, 1, memory_order_release);
  unsigned seen = 0;
  for (;;) {
    seen = wait_change(&team->generation, seen);
    if (atomic_load_explicit(&team->stop.value, memory_order_acquire))
      return NULL;
    team->fn(team->job, worker, team->size);
    atomic_fetch_add_explicit(&team->done.value, 1, memory_order_release);
  }
}

static Team *team_create(int index, int threads, const int *cpus) {
  Team *team = memory_alloc(1, sizeof(*team), MEM_METADATA);
  team->index = index;
  team->size = threads;
  team->cpus = cpus;
  team->assigned = memory_alloc((size_t)threads, sizeof(int), MEM_METADATA);
  team->observed = memory_alloc((size_t)threads, sizeof(int), MEM_METADATA);
  for (int w = 0; w < threads; w++) {
    team->assigned[w] = cpus ? cpus[w] : -1;
    team->observed[w] = -1;
  }
  team->threads = memory_alloc((size_t)threads, sizeof(pthread_t), MEM_METADATA);
  team->args = memory_alloc((size_t)threads, sizeof(*team->args), MEM_METADATA);
  if (cpus)
    team->observed[0] = pin_self(cpus[0]);
  for (int w = 1; w < threads; w++) {
    team->args[w].team = team;
    team->args[w].worker = w;
    int status = pthread_create(&team->threads[w], NULL, worker_main, &team->args[w]);
    if (status)
      fail("cannot create worker thread %d of team %d: %s", w, index, strerror(status));
  }
  wait_reach(&team->ready, (unsigned)(threads - 1));
  return team;
}

void pool_start(const PoolConfig *config) {
  if (team_count)
    fail("worker pool already started");
  if (config->teams < 1 || config->teams > 2)
    fail("invalid team count %d: expected 1 or 2", config->teams);
  for (int t = 0; t < config->teams; t++)
    if (config->threads[t] < 1 || config->threads[t] > TEAM_SUMMARY_THREADS)
      fail("invalid thread count %d for team %d", config->threads[t], t);
  for (int t = 0; t < config->teams; t++)
    teams[t] = team_create(t, config->threads[t], config->cpus[t]);
  team_count = summary_count = config->teams;
  for (int t = 0; t < team_count; t++) {
    TeamSummary *summary = &summaries[t];
    *summary = (TeamSummary){.index = t,
                             .threads = teams[t]->size,
                             .pinned = config->cpus[t] != NULL,
                             .verified = team_affinity_verified(teams[t])};
    memcpy(summary->assigned, teams[t]->assigned, (size_t)teams[t]->size * sizeof(int));
    memcpy(summary->observed, teams[t]->observed, (size_t)teams[t]->size * sizeof(int));
  }
}

void pool_stop(void) {
  if (in_pool_thread) /* a worker cannot join itself or its peers, e.g. from an exit handler after fail() */
    return;
  for (int t = 0; t < team_count; t++) {
    Team *team = teams[t];
    atomic_store_explicit(&team->stop.value, 1, memory_order_release);
    atomic_store_explicit(&team->generation.value, ++team->published, memory_order_release);
    for (int w = 1; w < team->size; w++)
      pthread_join(team->threads[w], NULL);
    memory_free(team->args);
    memory_free(team->threads);
    memory_free(team->observed);
    memory_free(team->assigned);
    memory_free(team);
    teams[t] = NULL;
  }
  team_count = 0;
}

Team *pool_team(int index) {
  return index >= 0 && index < team_count ? teams[index] : NULL;
}
int team_size(const Team *team) {
  return team->size;
}
int team_index(const Team *team) {
  return team->index;
}

void team_run(Team *team, TeamFn fn, const void *job) {
  if (team->size == 1) {
    fn(job, 0, 1);
    return;
  }
  team->fn = fn;
  team->job = job;
  atomic_store_explicit(&team->done.value, 0, memory_order_relaxed);
  atomic_store_explicit(&team->generation.value, ++team->published, memory_order_release);
  fn(job, 0, team->size);
  wait_reach(&team->done, (unsigned)(team->size - 1));
}

typedef struct {
  int rows;
  RowFn fn;
  const void *job;
} RowsJob;

static void rows_fn(const void *p, int worker, int workers) {
  const RowsJob *job = p;
  int r0 = (int)((long long)job->rows * worker / workers), r1 = (int)((long long)job->rows * (worker + 1) / workers);
  if (r1 > r0)
    job->fn(job->job, r0, r1);
}

void team_parallel_rows(Team *team, int rows, RowFn fn, const void *job) {
  RowsJob rows_job = {rows, fn, job};
  team_run(team, rows_fn, &rows_job);
}

void team_barrier(Team *team) {
  if (team->size == 1)
    return;
  unsigned generation = atomic_load_explicit(&team->barrier_generation.value, memory_order_acquire);
  if (atomic_fetch_add_explicit(&team->barrier_count.value, 1, memory_order_acq_rel) + 1 == (unsigned)team->size) {
    atomic_store_explicit(&team->barrier_count.value, 0, memory_order_relaxed);
    atomic_store_explicit(&team->barrier_generation.value, generation + 1, memory_order_release);
  } else {
    for (unsigned spins = 0; atomic_load_explicit(&team->barrier_generation.value, memory_order_acquire) == generation;
         spins++) {
      if (workers_aborted())
        return;
      if (spins < SPIN_LIMIT)
        relax();
      else
        sched_yield();
    }
  }
}

int team_worker_cpu(const Team *team, int worker) {
  return worker >= 0 && worker < team->size ? team->observed[worker] : -1;
}

int team_affinity_verified(const Team *team) {
  if (!team->cpus)
    return 0;
  for (int w = 0; w < team->size; w++)
    if (team->observed[w] != team->assigned[w])
      return 0;
  return 1;
}

int pool_summary_count(void) {
  return summary_count;
}
const TeamSummary *pool_summary(int index) {
  return index >= 0 && index < summary_count ? &summaries[index] : NULL;
}

/* ---- Cross-socket reduction (B4) -------------------------------------------------------------------------------- */

struct Exchange {
  int elements_max;
  size_t stride; /* floats between the two buffers of a socket, a multiple of a cache line */
  Team *team[2];
  float *buffer[2][2]; /* [socket][generation & 1] */
  void *heap[2];       /* NULL when the socket's buffers sit in a caller-owned region */
  Counter ready[2];    /* ready[s] = g: socket s finished writing partial g */
  Counter consumed[2]; /* consumed[s] = g: socket s finished reading the peer's partial g */
  unsigned issued[2];  /* generation handed out by exchange_partial; written by the coordinator, read by workers */
  atomic_int timing;
  double reduce_seconds[2], wait_seconds[2]; /* leader of each socket only */
};

size_t exchange_region_bytes(int elements_max) {
  size_t line = 64 / sizeof(float);
  size_t stride = (((size_t)elements_max + line - 1) / line) * line;
  return 2 * stride * sizeof(float);
}

Exchange *exchange_create(int elements_max, NumaRegion *const socket_regions[2]) {
  if (elements_max < 1)
    fail("invalid exchange size %d", elements_max);
  if (team_count != 2)
    fail("the cross-socket exchange needs a pool of two teams");
  Exchange *e = memory_alloc(1, sizeof(*e), MEM_METADATA);
  e->elements_max = elements_max;
  size_t bytes = exchange_region_bytes(elements_max);
  e->stride = bytes / (2 * sizeof(float));
  for (int s = 0; s < 2; s++) {
    e->team[s] = teams[s];
    char *base;
    NumaRegion *region = socket_regions ? socket_regions[s] : NULL;
    if (region) {
      if (!region->data || region->data_bytes < bytes)
        fail("exchange region of socket %d holds %zu bytes, %zu needed", s, region->data ? region->data_bytes : 0,
             bytes);
      base = region->data;
    } else {
      e->heap[s] = memory_alloc(bytes + 64, 1, MEM_ACTIVATION);
      base = (char *)(((uintptr_t)e->heap[s] + 63) & ~(uintptr_t)63);
    }
    if ((uintptr_t)base % 64)
      fail("exchange buffers of socket %d are not cache-line aligned", s);
    e->buffer[s][0] = (float *)base;
    e->buffer[s][1] = e->buffer[s][0] + e->stride;
  }
  return e;
}

/* Spins until *counter >= target (modular), or the abort flag; returns 0 on abort. */
static int wait_at_least(const Counter *counter, unsigned target) {
  for (unsigned spins = 0;; spins++) {
    if ((int)(atomic_load_explicit(&counter->value, memory_order_acquire) - target) >= 0)
      return 1;
    if (workers_aborted())
      return 0;
    if (spins < SPIN_LIMIT)
      relax();
    else
      sched_yield();
  }
}

void exchange_destroy(Exchange *e) {
  if (!e)
    return;
  for (int s = 0; s < 2; s++)
    wait_at_least(&e->consumed[s], e->issued[s]); /* the peer may still be reading our last partial */
  for (int s = 0; s < 2; s++)
    memory_free(e->heap[s]);
  memory_free(e);
}

static int exchange_check_socket(const Exchange *e, int socket) {
  if (!e || socket < 0 || socket > 1)
    fail("invalid exchange or socket %d", socket);
  return 1 - socket;
}

float *exchange_partial(Exchange *e, int socket) {
  int peer = exchange_check_socket(e, socket);
  double start = atomic_load_explicit(&e->timing, memory_order_relaxed) ? monotonic_seconds() : 0.0;
  unsigned g = ++e->issued[socket];
  /* The buffer's previous occupant is generation g - 2; the peer must have consumed it. */
  if (g > 2 && !wait_at_least(&e->consumed[peer], g - 2))
    return NULL;
  if (start)
    e->wait_seconds[socket] += monotonic_seconds() - start;
  return e->buffer[socket][g & 1];
}

void exchange_reduce_add(Exchange *e, int socket, int worker, int workers, float *residual, int elements) {
  int peer = exchange_check_socket(e, socket);
  if (workers != team_size(e->team[socket]) || worker < 0 || worker >= workers || elements < 0 ||
      elements > e->elements_max)
    fail("invalid exchange reduction: worker %d of %d, %d elements (maximum %d)", worker, workers, elements,
         e->elements_max);
  if (workers_aborted())
    return;
  unsigned g = e->issued[socket];
  int timed = worker == 0 && atomic_load_explicit(&e->timing, memory_order_relaxed);
  double t0 = timed ? monotonic_seconds() : 0.0;
  team_barrier(e->team[socket]); /* 1. every worker finished writing the local partial */
  if (worker == 0)
    atomic_store_explicit(&e->ready[socket].value, g, memory_order_release); /* 2 */
  if (!wait_at_least(&e->ready[peer], g))                                    /* 3 */
    return;
  double t1 = timed ? monotonic_seconds() : 0.0;
  const float *p0 = e->buffer[0][g & 1], *p1 = e->buffer[1][g & 1];
  int i0 = (int)((long long)elements * worker / workers), i1 = (int)((long long)elements * (worker + 1) / workers);
  for (int i = i0; i < i1; i++)
    residual[i] += p0[i] + p1[i];
  double t2 = timed ? monotonic_seconds() : 0.0;
  team_barrier(e->team[socket]); /* 4. every worker of this socket finished reading both partials */
  if (worker == 0)
    atomic_store_explicit(&e->consumed[socket].value, g, memory_order_release);
  if (timed) {
    e->wait_seconds[socket] += (t1 - t0) + (monotonic_seconds() - t2);
    e->reduce_seconds[socket] += t2 - t1;
  }
}

void exchange_abort(Exchange *e) {
  (void)e;
  workers_abort();
}

void exchange_set_timing(Exchange *e, int on) {
  atomic_store_explicit(&e->timing, on, memory_order_relaxed);
}

void exchange_timing(const Exchange *e, int socket, double *reduce_seconds, double *wait_seconds) {
  *reduce_seconds = e->reduce_seconds[socket];
  *wait_seconds = e->wait_seconds[socket];
}
