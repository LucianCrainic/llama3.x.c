#include "app.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

Options options = {.tokenizer = "tokenizer.bin",
                   .mode = "generate",
                   .steps = 4096,
                   .generation_length = -1,
                   .temperature = 1.0f,
                   .topp = 0.9f};
Model model;
static double program_start;
static char *owned_prompt;

static void usage(void) {
  fputs("Usage: run or runq CHECKPOINT [options]\n"
        "  -t FLOAT          temperature >= 0 (default 1; 0 = greedy)\n"
        "  -p FLOAT          top-p in [0,1] (default .9; 0 or 1 disables)\n"
        "  -s, --seed INT    nonnegative seed (0 maps to RNG state 1)\n"
        "  -n INT            maximum forward passes (default 4096; 0 = context)\n"
        "  -i, --prompt TEXT prompt (default empty; benchmark: Once upon a time)\n"
        "  --prompt-file PATH read UTF-8 prompt from file\n"
        "  -z PATH           matching tokenizer binary (default tokenizer.bin)\n"
        "  -m MODE           generate or chat\n"
        "  -y TEXT           chat system prompt\n"
        "  --benchmark       non-interactive; fixed default prompt and seed 1\n"
        "  --generation-length INT  cap generated tokens; 0 skips inference\n"
        "                    benchmark default: 128; -n/context still cap passes\n"
        "  --no-text         suppress generated text and prompt echo\n"
        "  --log-level LEVEL error|warn|info|debug|trace\n"
        "  --json PATH       write schema-v1 performance report to a file\n"
        "  -h, --help        show this help\n",
        stderr);
}
static unsigned long long integer(const char *value, const char *flag, unsigned long long max) {
  char *end;
  errno = 0;
  if (!*value || *value == '-' || *value == '+' || *value == ' ')
    fail("invalid integer for %s: '%s'", flag, value);
  unsigned long long result = strtoull(value, &end, 10);
  if (errno || *end || result > max)
    fail("invalid integer for %s: '%s'", flag, value);
  return result;
}
static float real_value(const char *value, const char *flag, float max) {
  char *end;
  errno = 0;
  float result = strtof(value, &end);
  if (!*value || errno || *end || !isfinite(result) || result < 0 || result > max)
    fail("invalid value for %s: '%s'", flag, value);
  return result;
}
static void parse(int argc, char **argv) {
  int seed_set = 0, steps_set = 0;
  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    usage();
    exit(0);
  }
  if (argc < 2) {
    usage();
    fail("checkpoint path is required");
  }
  options.checkpoint = argv[1];
  /* Find report destination before validating options so CLI failures can be recorded. */
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--benchmark") || !strcmp(argv[i], "--no-text") || !strcmp(argv[i], "--help") ||
        !strcmp(argv[i], "-h"))
      continue;
    if (!strcmp(argv[i], "--json") && i + 1 < argc)
      options.json = argv[i + 1];
    i++; /* Every remaining option consumes exactly one value. */
  }
  for (int i = 2; i < argc; i++) {
    const char *flag = argv[i];
    if (!strcmp(flag, "--benchmark")) {
      options.benchmark = 1;
      continue;
    }
    if (!strcmp(flag, "--no-text")) {
      options.no_text = 1;
      continue;
    }
    if (!strcmp(flag, "--help") || !strcmp(flag, "-h")) {
      options.json = NULL;
      usage();
      exit(0);
    }
    if (i + 1 == argc)
      fail("missing value for '%s'", flag);
    const char *value = argv[++i];
    if (!strcmp(flag, "-t"))
      options.temperature = real_value(value, flag, 1e6f);
    else if (!strcmp(flag, "-p"))
      options.topp = real_value(value, flag, 1);
    else if (!strcmp(flag, "-s") || !strcmp(flag, "--seed")) {
      options.seed = integer(value, flag, ULLONG_MAX);
      seed_set = 1;
    } else if (!strcmp(flag, "-n")) {
      options.steps = (int)integer(value, flag, INT_MAX);
      steps_set = 1;
    } else if (!strcmp(flag, "--generation-length"))
      options.generation_length = (int)integer(value, flag, INT_MAX);
    else if (!strcmp(flag, "-i") || !strcmp(flag, "--prompt"))
      options.prompt = value;
    else if (!strcmp(flag, "--prompt-file"))
      options.prompt_file = value;
    else if (!strcmp(flag, "-z"))
      options.tokenizer = value;
    else if (!strcmp(flag, "-m"))
      options.mode = value;
    else if (!strcmp(flag, "-y"))
      options.system_prompt = value;
    else if (!strcmp(flag, "--json"))
      options.json = value;
    else if (!strcmp(flag, "--log-level")) {
      const char *levels[] = {"error", "warn", "info", "debug", "trace"};
      int found = 0;
      for (int j = 0; j < 5; j++)
        if (!strcmp(value, levels[j])) {
          log_level = (LogLevel)j;
          found = 1;
        }
      if (!found)
        fail("invalid log level '%s'", value);
    } else
      fail("unknown argument '%s'", flag);
  }
  if (strcmp(options.mode, "generate") && strcmp(options.mode, "chat"))
    fail("unsupported mode '%s'", options.mode);
  if (options.prompt && options.prompt_file)
    fail("use either --prompt or --prompt-file");
  if (!strcmp(options.mode, "chat") && (options.benchmark || options.generation_length >= 0))
    fail("chat cannot be combined with benchmark or generation-length");
  if (options.benchmark) {
    if (!options.prompt && !options.prompt_file)
      options.prompt = "Once upon a time";
    if (options.generation_length < 0 && !steps_set)
      options.generation_length = 128;
  }
  if (!seed_set)
    options.seed = options.benchmark ? 1 : (unsigned long long)time(NULL);
  if (!options.seed)
    options.seed = 1;
}

/* Grow owned prompt buffers; the cap keeps token counts representable as int. */
static char *read_text(FILE *stream, int line, int *has_input) {
  size_t capacity = 256, n = 0;
  char *text = memory_alloc(capacity, 1, MEM_PROMPT);
  int c;
  *has_input = 0;
  while ((c = fgetc(stream)) != EOF) {
    *has_input = 1;
    if (line && c == '\n')
      break;
    if (c == 0)
      fail("prompt contains a NUL byte");
    if (n >= INT_MAX - 64)
      fail("prompt is too long");
    if (n + 1 == capacity) {
      size_t next = capacity * 2;
      char *grown = memory_alloc(next, 1, MEM_PROMPT);
      memcpy(grown, text, n);
      memory_free(text);
      text = grown;
      capacity = next;
    }
    text[n++] = (char)c;
  }
  if (ferror(stream))
    fail("cannot read prompt: %s", strerror(errno));
  text[n] = 0;
  return text;
}
static int *tokenize(Tokenizer *t, const char *text, int bos, int *n) {
  size_t bytes = strlen(text);
  if (bytes > INT_MAX - 64)
    fail("prompt is too long");
  int *tokens = memory_alloc(bytes + 2, sizeof(int), MEM_PROMPT);
  encode(t, text, (int8_t)bos, 0, tokens, n);
  return tokens;
}

/* Sequential prefill: every supplied token, including the last prompt token,
 * executes one forward pass. Only its final logits are sampled. A decode step
 * forwards the previously generated token. EOS/EOT are sampled but not emitted. */
static int run_sequence(Tokenizer *t, Sampler *sampler, const int *tokens, int n, int *position, int generation_limit,
                        int echo_prompt, double turn_start) {
  metrics.input_tokens += (size_t)n;
  LOG(LOG_TRACE, "sequence at position %d: %d prompt tokens", *position, n);
  if (generation_limit == 0) {
    metrics.stop_reason = "generation_limit";
    return -1;
  }
  float *logits = NULL;
  metrics.inference_started = 1;
  for (int i = 0; i < n && *position < options.steps; i++) {
    double start = monotonic_seconds();
    logits = engine_forward(&model, tokens[i], (*position)++);
    metrics.prefill_s += monotonic_seconds() - start;
    metrics.processed_tokens++;
    metrics.forward_passes++;
    if (echo_prompt && i + 1 < n && !options.no_text) {
      safe_printf(decode(t, tokens[i + 1]));
      fflush(stdout);
    }
    if (i + 1 < n && *position == options.steps) {
      metrics.stop_reason = "step_limit";
      return -1;
    }
  }
  if (!logits) {
    metrics.stop_reason = "step_limit";
    return -1;
  }
  int output_count = 0;
  for (;;) {
    int next = sample(sampler, logits);
    metrics.sampled_tokens++;
    if (next == EOS_ID || next == EOT_ID) {
      metrics.stop_tokens++;
      metrics.stop_reason = "stop_token";
      return next;
    }
    metrics.output_tokens++;
    output_count++;
    if (!metrics.has_first_token) {
      metrics.ttft_s = monotonic_seconds() - turn_start;
      metrics.has_first_token = 1;
    }
    if (!options.no_text) {
      safe_printf(decode(t, next));
      fflush(stdout);
    }
    if (generation_limit >= 0 && output_count >= generation_limit) {
      metrics.stop_reason = "generation_limit";
      break;
    }
    if (*position == options.steps) {
      metrics.stop_reason = "step_limit";
      break;
    }
    double start = monotonic_seconds();
    logits = engine_forward(&model, next, (*position)++);
    metrics.decode_s += monotonic_seconds() - start;
    metrics.decode_steps++;
    metrics.forward_passes++;
  }
  return -1;
}

static void generate(Tokenizer *t, Sampler *s, const char *prompt) {
  metrics.turns++;
  double start = monotonic_seconds();
  int n = 0;
  int *tokens = tokenize(t, prompt ? prompt : "", 1, &n);
  metrics.tokenization_s += monotonic_seconds() - start;
  int pos = 0;
  run_sequence(t, s, tokens, n, &pos, options.generation_length, 1, start);
  if (!options.no_text)
    putchar('\n');
  metrics.inference_s += monotonic_seconds() - start;
  memory_free(tokens);
}

static char *input_line(const char *guide, int *ok) {
  fputs(guide, stderr);
  fflush(stderr);
  return read_text(stdin, 1, ok);
}
static void chat(Tokenizer *t, Sampler *s) {
  if (t->vocab_size <= EOT_ID)
    fail("chat requires the original LLaMA 3 vocabulary");
  int ok = 1, pos = 0;
  char *owned_system = NULL;
  const char *system = options.system_prompt;
  if (!system) {
    owned_system = input_line("Enter system prompt (optional): ", &ok);
    system = owned_system;
  }
  while (pos < options.steps && ok) {
    char *owned_user = NULL;
    const char *user = (pos == 0 ? options.prompt : NULL);
    if (!user) {
      owned_user = input_line("User (or exit): ", &ok);
      user = owned_user;
    }
    if (!ok || !strcmp(user, "exit")) {
      memory_free(owned_user);
      metrics.stop_reason = "user_exit";
      break;
    }
    metrics.turns++;
    double start = monotonic_seconds();
    int ns = 0, nu = 0;
    int *st = NULL;
    if (pos == 0)
      st = tokenize(t, system, 0, &ns);
    int *ut = tokenize(t, user, 0, &nu);
    size_t count = checked_add(checked_add((size_t)ns, (size_t)nu), 32);
    if (count > INT_MAX)
      fail("chat prompt is too long");
    int *tokens = memory_alloc(count, sizeof(int), MEM_PROMPT);
    int n = 0;
    if (pos == 0) {
      const int header[] = {BOS_ID, 128006, 9125, 128007, 271};
      for (size_t i = 0; i < 5; i++)
        tokens[n++] = header[i];
      for (int i = 0; i < ns; i++)
        tokens[n++] = st[i];
      tokens[n++] = EOT_ID;
    }
    const int uh[] = {128006, 882, 128007, 271};
    for (size_t i = 0; i < 4; i++)
      tokens[n++] = uh[i];
    for (int i = 0; i < nu; i++)
      tokens[n++] = ut[i];
    const int ah[] = {EOT_ID, 128006, 78191, 128007, 271};
    for (size_t i = 0; i < 5; i++)
      tokens[n++] = ah[i];
    metrics.tokenization_s += monotonic_seconds() - start;
    fputs("Assistant: ", stderr);
    int stop = run_sequence(t, s, tokens, n, &pos, -1, 0, start);
    if (!options.no_text)
      putchar('\n');
    metrics.inference_s += monotonic_seconds() - start;
    /* Cache the end-of-turn marker before consuming the next user template. */
    if (stop >= 0 && pos < options.steps) {
      double tick = monotonic_seconds();
      engine_forward(&model, stop, pos++);
      double elapsed = monotonic_seconds() - tick;
      metrics.decode_s += elapsed;
      metrics.inference_s += elapsed;
      metrics.decode_steps++;
      metrics.forward_passes++;
    }
    memory_free(tokens);
    memory_free(st);
    memory_free(ut);
    memory_free(owned_user);
  }
  memory_free(owned_system);
}

static void finish(void) {
  metrics.end_to_end_s = monotonic_seconds() - program_start;
  write_report();
  memory_free(owned_prompt);
}
int main(int argc, char **argv) {
  program_start = monotonic_seconds();
  atexit(finish);
  parse(argc, argv);
  if (options.prompt_file) {
    FILE *file = fopen(options.prompt_file, "rb");
    if (!file)
      fail("cannot open prompt file '%s': %s", options.prompt_file, strerror(errno));
    int ok;
    owned_prompt = read_text(file, 0, &ok);
    fclose(file);
    options.prompt = owned_prompt;
  }
  engine_load(&model, options.checkpoint);
  if (!options.steps || options.steps > model.config.seq_len)
    options.steps = model.config.seq_len;
  Tokenizer t = {0};
  double start = monotonic_seconds();
  build_tokenizer(&t, options.tokenizer, model.config.vocab_size);
  metrics.tokenizer_s = monotonic_seconds() - start;
  metrics.tokenizer_ready = 1;
  Sampler s = {0};
  start = monotonic_seconds();
  build_sampler(&s, model.config.vocab_size, options.temperature, options.topp, options.seed);
  metrics.runtime_s += monotonic_seconds() - start;
  LOG(LOG_INFO, "%s: dim=%d layers=%d context=%d; mapped=%zu bytes; seed=%llu", engine_precision(), model.config.dim,
      model.config.n_layers, model.config.seq_len, model.mapped_bytes, options.seed);
  if (!strcmp(options.mode, "chat"))
    chat(&t, &s);
  else
    generate(&t, &s, options.prompt);
  if (!options.no_text && fflush(stdout))
    fail("cannot flush generated text: %s", strerror(errno));
  if (ferror(stdout))
    fail("cannot write generated text: %s", strerror(errno));
  LOG(LOG_INFO,
      "prompt=%zu processed=%zu output=%zu forwards=%zu decode_steps=%zu; prefill=%.6fs decode=%.6fs inference=%.6fs",
      metrics.input_tokens, metrics.processed_tokens, metrics.output_tokens, metrics.forward_passes,
      metrics.decode_steps, metrics.prefill_s, metrics.decode_s, metrics.inference_s);
  free_sampler(&s);
  free_tokenizer(&t);
  engine_free(&model);
  metrics.completed = 1;
  return 0;
}
