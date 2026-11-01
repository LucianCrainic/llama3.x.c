// Exact-prompt CPU benchmark using the public llama.cpp API.
// argv: model.gguf rendered-prompt.txt expected-ids.txt threads batch output-limit
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <sched.h>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char **argv) try {
    check(argc == 7, "usage: model prompt expected-ids threads batch output-limit");
    const int threads = std::stoi(argv[4]), batch_size = std::stoi(argv[5]);
    const int limit = std::stoi(argv[6]);
    check(threads > 0 && batch_size > 0 && limit > 0, "invalid dimensions");
    std::ifstream prompt_file(argv[2]), id_file(argv[3]);
    check(prompt_file.good() && id_file.good(), "cannot read prompt files");
    const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
    std::vector<llama_token> expected;
    for (int id; id_file >> id;) expected.push_back(id);
    check(!expected.empty() && expected.size() + limit <= 2048, "invalid prompt length");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    auto *model = llama_model_load_from_file(argv[1], mp);
    check(model != nullptr, "model load failed");
    const auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 8);
    const int n = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    check(n > 0, "tokenization failed");
    tokens.resize(n);
    check(tokens == expected, "llama.cpp/HF/C prompt token IDs differ");
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = batch_size;
    cp.n_ubatch = batch_size;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.no_perf = false;
    auto *ctx = llama_init_from_model(model, cp);
    check(ctx != nullptr, "context init failed");
    auto batch = llama_batch_init(batch_size, 0, 1);
    auto forward = [&](llama_token *input, int count, int position) {
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = input[i];
            batch.pos[i] = position + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            // Sequential mode computes logits at every prompt position, like C.
            batch.logits[i] = (i == count - 1);
        }
        check(llama_decode(ctx, batch) == 0, "llama_decode failed");
        llama_synchronize(ctx);
    };
    double prefill = 0, decode = 0;
    for (int position = 0; position < n;) {
        const int count = std::min(batch_size, n - position);
        const auto start = Clock::now();
        forward(tokens.data() + position, count, position);
        prefill += seconds(start);
        position += count;
    }
    std::vector<llama_token> generated;
    int decode_steps = 0;
    bool stopped = false;
    const int nv = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < limit; ++i) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        check(logits != nullptr, "missing logits");
        llama_token token = std::max_element(logits, logits + nv) - logits;
        if (llama_vocab_is_eog(vocab, token)) { stopped = true; break; }
        generated.push_back(token);
        if (i + 1 < limit) {
            const auto start = Clock::now();
            forward(&token, 1, n + i);
            decode += seconds(start);
            ++decode_steps;
        }
    }
    struct rusage usage{};
    check(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    check(sched_getaffinity(0, sizeof(affinity), &affinity) == 0, "getaffinity failed");
    std::printf("{\"input\":%d,\"output\":%zu,\"decode_steps\":%d,\"stopped_eog\":%s,"
                "\"threads\":%d,\"batch\":%d,\"context\":2048,\"weights\":\"f32\",\"kv\":\"f32\","
                "\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,"
                "\"prefill_tok_s\":%.9f,\"decode_tok_s\":%.9f,\"peak_rss_bytes\":%lld,"
                "\"affinity_cpus\":[", n, generated.size(), decode_steps, stopped ? "true" : "false",
                threads, batch_size, prefill, decode, n/prefill, decode_steps/decode,
                static_cast<long long>(usage.ru_maxrss)*1024);
    bool comma = false;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &affinity)) {
        std::printf("%s%d", comma ? "," : "", i); comma = true;
    }
    std::printf("],\"generated_ids\":[");
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", generated[i]);
    std::printf("]}\n");
    llama_perf_context_print(ctx);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
