CC ?= cc
MODE ?= release
OPENMP ?= 0
NATIVE ?= 0
PCRE2 ?= 1
BUILD_DIR ?= build/$(MODE)
CPPFLAGS += -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -D_DARWIN_C_SOURCE -D_FILE_OFFSET_BITS=64 -Iinclude
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -pthread
LDLIBS += -lm -pthread

ifeq ($(MODE),release)
CFLAGS += -O3
else ifeq ($(MODE),debug)
CFLAGS += -O0 -g3
else ifeq ($(MODE),profile)
CFLAGS += -O2 -g -fno-omit-frame-pointer
else ifeq ($(MODE),sanitize)
CFLAGS += -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
LDFLAGS += -fsanitize=address,undefined
else ifeq ($(MODE),tsan)
CFLAGS += -O1 -g -fno-omit-frame-pointer -fsanitize=thread
LDFLAGS += -fsanitize=thread
else
$(error MODE must be release, debug, profile, sanitize, or tsan)
endif
ifeq ($(OPENMP),1)
CFLAGS += $(if $(OMP_CFLAGS),$(OMP_CFLAGS),-fopenmp)
LDFLAGS += $(if $(OMP_LDFLAGS),$(OMP_LDFLAGS),-fopenmp)
endif
ifeq ($(NATIVE),1)
CFLAGS += -march=native
endif
ifeq ($(PCRE2),1)
CPPFLAGS += -DLLAMA_PCRE2 $(shell pkg-config --cflags libpcre2-8)
LDLIBS += $(shell pkg-config --libs libpcre2-8)
endif
COMMON = src/runtime.c src/kernels.c src/kernels_simd.c src/profile.c src/numa.c src/shards.c src/workers.c src/tp.c src/gemm.c src/loader.c src/prefill.c src/tokenizer.c src/sampler.c
APP = src/app.c src/report.c
HEADERS = include/runtime.h include/app.h include/kernels.h include/profile.h include/numa.h include/shards.h include/workers.h include/tp.h include/gemm.h include/loader.h include/prefill.h
# Regenerate the configuration stamp only when flags change; changing OPENMP or
# CC must not silently reuse an executable from an earlier build.
.PHONY: all clean test FORCE
all: $(BUILD_DIR)/run $(BUILD_DIR)/runq
$(BUILD_DIR):
	mkdir -p $@
$(BUILD_DIR)/config: FORCE | $(BUILD_DIR)
	@printf '%s\n' '$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(LDLIBS)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@
$(BUILD_DIR)/run: src/run.c $(COMMON) $(APP) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DBUILD_MODE='"$(MODE)"' -DBUILD_FLAGS='"$(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(LDLIBS)"' $(CFLAGS) src/run.c $(COMMON) $(APP) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/runq: src/runq.c $(COMMON) $(APP) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DBUILD_MODE='"$(MODE)"' -DBUILD_FLAGS='"$(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(LDLIBS)"' $(CFLAGS) src/runq.c $(COMMON) $(APP) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-fp32: tests/unit.c src/run.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/unit.c src/run.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-int8: tests/unit.c src/runq.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/unit.c src/runq.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-kernels: tests/kernels.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/kernels.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-gemm: tests/gemm.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/gemm.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
# The same test with the AVX-512 microkernel running on a scalar emulation, so its code is exercised on any host.
$(BUILD_DIR)/test-gemm-emul: tests/gemm.c tests/avx512_emul.h $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DGEMM_EMULATE_AVX512 $(CFLAGS) tests/gemm.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-amx: tests/amx.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/amx.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
# The same test with the AMX tile instructions replaced by the software model of tests/amx_emul.h.
$(BUILD_DIR)/test-amx-emul: tests/amx.c tests/amx_emul.h $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DGEMM_EMULATE_AMX $(CFLAGS) tests/amx.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-bf16: tests/bf16.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/bf16.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-bf16-emul: tests/bf16.c tests/amx_emul.h $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DGEMM_EMULATE_AMX $(CFLAGS) tests/bf16.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-bf16-engine: tests/bf16_engine.c src/run.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/bf16_engine.c src/run.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-bf16-engine-emul: tests/bf16_engine.c src/run.c tests/amx_emul.h $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DGEMM_EMULATE_AMX $(CFLAGS) tests/bf16_engine.c src/run.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
# The CLI with the AMX tile instructions on the software model, for the report, mode-guard and fixture tests.
$(BUILD_DIR)/run-amx-emul: src/run.c tests/amx_emul.h $(COMMON) $(APP) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DGEMM_EMULATE_AMX -DBUILD_MODE='"$(MODE)"' -DBUILD_FLAGS='"emulated AMX"' $(CFLAGS) src/run.c $(COMMON) $(APP) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-q8: tests/q8.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/q8.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-q8-emul: tests/q8.c tests/amx_emul.h tests/avx512_emul.h $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DGEMM_EMULATE_AMX -DGEMM_EMULATE_AVX512 $(CFLAGS) tests/q8.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-q8-engine: tests/q8_engine.c src/runq.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/q8_engine.c src/runq.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-q8-engine-emul: tests/q8_engine.c src/runq.c tests/amx_emul.h $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DGEMM_EMULATE_AMX $(CFLAGS) tests/q8_engine.c src/runq.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/runq-amx-emul: src/runq.c tests/amx_emul.h $(COMMON) $(APP) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) -DGEMM_EMULATE_AMX -DBUILD_MODE='"$(MODE)"' -DBUILD_FLAGS='"emulated AMX"' $(CFLAGS) src/runq.c $(COMMON) $(APP) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/amx-bench: tests/amx_bench.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/amx_bench.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/gemm-bench: tests/gemm_bench.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/gemm_bench.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-manifest: tests/manifest.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/manifest.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-workers: tests/workers.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/workers.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/stream-bench: tests/stream_bench.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/stream_bench.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-shards: tests/shards.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/shards.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-regions: tests/regions.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/regions.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/numa_probe: tests/numa_probe.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/numa_probe.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-prefill-fp32: tests/prefill.c src/run.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/prefill.c src/run.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test-prefill-int8: tests/prefill.c src/runq.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/prefill.c src/runq.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
test: all $(BUILD_DIR)/test-fp32 $(BUILD_DIR)/test-int8 $(BUILD_DIR)/test-kernels $(BUILD_DIR)/test-gemm $(BUILD_DIR)/test-gemm-emul $(BUILD_DIR)/test-amx $(BUILD_DIR)/test-amx-emul $(BUILD_DIR)/test-bf16 $(BUILD_DIR)/test-bf16-emul $(BUILD_DIR)/amx-bench $(BUILD_DIR)/test-q8 $(BUILD_DIR)/test-q8-emul $(BUILD_DIR)/test-q8-engine $(BUILD_DIR)/test-q8-engine-emul $(BUILD_DIR)/runq-amx-emul $(BUILD_DIR)/test-bf16-engine $(BUILD_DIR)/test-bf16-engine-emul $(BUILD_DIR)/run-amx-emul $(BUILD_DIR)/gemm-bench $(BUILD_DIR)/test-manifest $(BUILD_DIR)/test-shards $(BUILD_DIR)/numa_probe $(BUILD_DIR)/test-regions $(BUILD_DIR)/test-workers $(BUILD_DIR)/test-prefill-fp32 $(BUILD_DIR)/test-prefill-int8 $(BUILD_DIR)/probe-fp32 $(BUILD_DIR)/probe-int8
	$(BUILD_DIR)/test-kernels
	$(BUILD_DIR)/test-gemm
	$(BUILD_DIR)/test-gemm-emul
	$(BUILD_DIR)/test-amx
	$(BUILD_DIR)/test-amx-emul
	$(BUILD_DIR)/test-bf16
	$(BUILD_DIR)/test-bf16-emul
	$(BUILD_DIR)/test-q8
	$(BUILD_DIR)/test-q8-emul
	$(BUILD_DIR)/test-manifest
	$(BUILD_DIR)/test-shards
	$(BUILD_DIR)/test-regions
	$(BUILD_DIR)/test-workers
	python3 tests/test_runtime.py $(BUILD_DIR)
	python3 tests/test_numa.py $(BUILD_DIR)
	python3 tests/test_tp.py $(BUILD_DIR)
	BUILD_DIR=$(BUILD_DIR) python3 tests/test_tools.py
clean:
	rm -rf build

$(BUILD_DIR)/probe-fp32: tests/probe.c src/run.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/probe.c src/run.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/probe-int8: tests/probe.c src/runq.c $(COMMON) $(HEADERS) $(BUILD_DIR)/config
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/probe.c src/runq.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: test-reference
test-reference: all $(BUILD_DIR)/probe-fp32 $(BUILD_DIR)/probe-int8
	$(if $(PYTHON),$(PYTHON),python3) tests/test_reference.py $(BUILD_DIR)
