#include "runtime.h"
#include "bench.h"

static const float *forward(void *ctx, int *ids, int count, int pos, int logits) {
  Model *model = ctx;
  if (count == 1 && logits)
    return engine_forward(model, ids[0], pos);
  return engine_forward_batch(model, ids, count, pos, logits ? LOGITS_LAST : LOGITS_NONE);
}

int main(int argc, char **argv) {
  if (argc != 6)
    return 2;
  Model model = {0};
  int capacity = atoi(argv[4]), batch = atoi(argv[5]);
  if (capacity <= 0 || batch <= 0)
    return 2;
  engine_load_context(&model, argv[1], capacity);
  if (model.config.seq_len != capacity || engine_max_batch(&model) != batch)
    fail("Native benchmark context or batch differs from the requested setup");
  Bench bench = {&model, forward, NULL, model.config.vocab_size, batch, capacity};
  requests(&bench, "short", argv[2]);
  int position = requests(&bench, "long", argv[3]);
  KvStats kv = {0};
  engine_kv_stats(&model, position, 1, &kv);
  if (!kv.verified || kv.pages_wrong_node || kv.pages_missing)
    fail("Active KV page verification failed");
  fprintf(stderr, "Verified %zu active KV bytes at %d positions (%zu page queries)\n",
          kv.active_bytes, position, kv.pages_queried);
  engine_free(&model);
  return 0;
}
