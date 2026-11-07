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
    wait_change(&team->barrier_generation, generation);
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
