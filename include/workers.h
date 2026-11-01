#ifndef LLAMA_WORKERS_H
#define LLAMA_WORKERS_H

/* Persistent worker teams: the only scheduler. Threads are created once (pool_start) and wait for jobs; the caller
 * of team_run is worker 0 of its team and does its share instead of sleeping. Workers synchronize with a
 * generation-counted barrier built on C11 atomics: they spin for a bounded time, then sched_yield. Workers must not
 * touch the engine's global accounting (memory_alloc, metrics); only the coordinator thread does. */
typedef struct Team Team;
typedef void (*RowFn)(const void *job, int r0, int r1);
typedef void (*TeamFn)(const void *job, int worker, int workers);

typedef struct {
  int teams;          /* 1 or 2 */
  int threads[2];     /* per team, leader included */
  const int *cpus[2]; /* one CPU id per thread, in worker order; NULL = unpinned */
} PoolConfig;

void pool_start(const PoolConfig *config); /* once, at load; fails on a bad configuration */
void pool_stop(void);                      /* joins every worker; a no-op when called from a pool thread */
Team *pool_team(int index);                /* NULL when not started */
int team_size(const Team *team);
int team_index(const Team *team); /* socket/team id */

/* Runs fn(job, worker, workers) on every member and returns after all have finished. Not reentrant. */
void team_run(Team *team, TeamFn fn, const void *job);
/* Splits [0, rows) into one contiguous, balanced range per member. */
void team_parallel_rows(Team *team, int rows, RowFn fn, const void *job);
/* Barrier among all members; only valid inside a TeamFn. */
void team_barrier(Team *team);

/* Pinning evidence. team_worker_cpu is the CPU the worker's affinity mask was read back as (-1 when it is not a
 * single CPU, or unpinned); team_affinity_verified is 1 when every worker was pinned and read back as assigned. */
int team_worker_cpu(const Team *team, int worker);

/* What the report records about each team. It is filled at pool_start and survives pool_stop. */
enum { TEAM_SUMMARY_THREADS = 1024 };
typedef struct {
  int index, threads, pinned, verified;
  int assigned[TEAM_SUMMARY_THREADS], observed[TEAM_SUMMARY_THREADS]; /* CPU ids; -1 unpinned or unknown */
} TeamSummary;
int pool_summary_count(void);
const TeamSummary *pool_summary(int index);
int team_affinity_verified(const Team *team);

#endif
