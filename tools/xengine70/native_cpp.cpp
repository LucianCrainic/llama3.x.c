#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <sched.h>
#include "bench.h"

static const float *forward(void *ctx, int *ids, int count, int, int want_logits) {
  auto *context = static_cast<llama_context *>(ctx);
  static int8_t logits[256] = {};
  if (count > int(sizeof(logits)))
    exit(2);
  auto batch = llama_batch_get_one(ids, count);
  logits[count - 1] = want_logits;
  batch.logits = logits;
  int status = llama_decode(context, batch);
  logits[count - 1] = 0;
  if (status)
    exit(2);
  llama_synchronize(context);
  return want_logits ? llama_get_logits_ith(context, -1) : nullptr;
}

static void reset(void *ctx) {
  llama_memory_clear(llama_get_memory(static_cast<llama_context *>(ctx)), false);
}

int main(int argc, char **argv) {
  if (argc != 7)
    return 2;
  int capacity = atoi(argv[4]), batch = atoi(argv[5]), threads = atoi(argv[6]);
  if (capacity <= 0 || batch <= 0 || batch > 256 || threads <= 0)
    return 2;
  ggml_backend_load_all();
  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_NUMACTL);
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  mp.load_mode = LLAMA_LOAD_MODE_NONE;
  auto *model = llama_model_load_from_file(argv[1], mp);
  if (!model)
    return 2;
  auto p = llama_context_default_params();
  p.n_ctx = capacity;
  p.n_batch = p.n_ubatch = batch;
  p.n_threads = p.n_threads_batch = threads;
  p.type_k = p.type_v = GGML_TYPE_F32;
  auto *ctx = llama_init_from_model(model, p);
  if (!ctx || int(llama_n_ctx(ctx)) != capacity || int(llama_n_batch(ctx)) != batch)
    return 2;
  auto tpp = ggml_threadpool_params_default(threads);
  cpu_set_t cpus;
  if (sched_getaffinity(0, sizeof(cpus), &cpus) || CPU_COUNT(&cpus) != threads)
    return 2;
  for (int cpu = 0; cpu < GGML_MAX_N_THREADS && cpu < CPU_SETSIZE; ++cpu)
    tpp.cpumask[cpu] = CPU_ISSET(cpu, &cpus);
  tpp.strict_cpu = true;
  auto *pool = ggml_threadpool_new(&tpp);
  if (!pool)
    return 2;
  llama_attach_threadpool(ctx, pool, nullptr);
  Bench bench = {ctx, forward, reset, llama_vocab_n_tokens(llama_model_get_vocab(model)), batch, capacity};
  requests(&bench, "short", argv[2]);
  requests(&bench, "long", argv[3]);
  llama_detach_threadpool(ctx);
  ggml_threadpool_free(pool);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
}
