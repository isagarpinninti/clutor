# Builds the Clutor engine. Needs only a C11 compiler and pthreads.
CC ?= cc
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter
ifeq ($(shell uname -m),arm64)
  CFLAGS += -mcpu=native
endif
ifeq ($(shell uname -m),x86_64)
  CFLAGS += -mavx2
endif
LDLIBS = -lm -lpthread
# BUILD can point somewhere writable when the source sits in a read-only installed package: make BUILD=$HOME/.clutor/bin
BUILD ?= build
SRC = src/gguf.c src/cache.c src/route.c src/kernel.c src/quant.c src/meta.c src/xs.c src/forward.c src/tokenizer.c src/stream.c
BIN = $(BUILD)/clutor-engine

all: $(BIN)
build:
	mkdir -p build
$(BIN): $(SRC) src/main.c $(wildcard src/*.h)
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SRC) src/main.c -o $@ $(LDLIBS)
build/test_engine: build $(SRC) test/test_engine.c $(wildcard src/*.h)
	$(CC) $(CFLAGS) -Isrc $(SRC) test/test_engine.c -o $@ $(LDLIBS)
FIXTURE = ../packages/core/test/fixtures/stories260K.gguf
test: build/test_engine
	./build/test_engine $(FIXTURE)

# The kernel has one source and several compile-time paths. `make test` runs the one this machine picks; this builds and runs the others:
# portable C (everywhere), NEON without the dot-product instruction (arm64), and AVX2 cross-built for x86_64 and run through Rosetta if it can.
TEST_SRC = $(SRC) test/test_engine.c
PATH_FLAGS = -O2 -g -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter -Isrc
test-paths: build
	$(CC) $(PATH_FLAGS) -DCL_FORCE_SCALAR $(TEST_SRC) -o build/test-scalar $(LDLIBS)
	./build/test-scalar $(FIXTURE)
ifeq ($(shell uname -m),arm64)
	$(CC) $(PATH_FLAGS) -mcpu=native -DCL_NO_DOTPROD $(TEST_SRC) -o build/test-neon $(LDLIBS)
	./build/test-neon $(FIXTURE)
endif
test-avx2: build
	$(CC) -arch x86_64 $(PATH_FLAGS) -mavx2 $(TEST_SRC) -o build/test-avx2 $(LDLIBS)
	arch -x86_64 ./build/test-avx2 $(FIXTURE)

# Golden test: a tiny generated OLMoE with random K-quant weights, checked against an independent pure-Python forward pass (needs only python3).
# Run on the default build and, to cover every kernel path, on the portable-C and NEON-without-dotprod builds.
GOLDEN_GGUF = build/tiny_olmoe.gguf
GOLDEN_Q3 = build/tiny_qwen3moe.gguf
golden-files: build
	python3 test/tiny_olmoe.py write $(GOLDEN_GGUF)
	python3 test/tiny_olmoe.py write $(GOLDEN_Q3) --arch qwen3moe
golden: $(BIN) golden-files
	python3 test/tiny_olmoe.py check $(BIN) $(GOLDEN_GGUF)
	python3 test/tiny_olmoe.py check $(BIN) $(GOLDEN_Q3) --arch qwen3moe
golden-paths: $(BIN) golden-files
	python3 test/tiny_olmoe.py check $(BIN) $(GOLDEN_GGUF)
	python3 test/tiny_olmoe.py check $(BIN) $(GOLDEN_Q3) --arch qwen3moe
	$(CC) $(PATH_FLAGS) -DCL_FORCE_SCALAR $(SRC) src/main.c -o build/engine-scalar $(LDLIBS)
	python3 test/tiny_olmoe.py check build/engine-scalar $(GOLDEN_GGUF)
	python3 test/tiny_olmoe.py check build/engine-scalar $(GOLDEN_Q3) --arch qwen3moe
ifeq ($(shell uname -m),arm64)
	$(CC) $(PATH_FLAGS) -mcpu=native -DCL_NO_DOTPROD $(SRC) src/main.c -o build/engine-neon $(LDLIBS)
	python3 test/tiny_olmoe.py check build/engine-neon $(GOLDEN_GGUF)
	python3 test/tiny_olmoe.py check build/engine-neon $(GOLDEN_Q3) --arch qwen3moe
endif

# Mutation fuzzing of the file readers under ASan + UBSan: corrupt a valid model many ways, expect clean errors and never a crash or hang.
fuzz: build
	$(CC) -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer $(SRC) src/main.c -o build/engine-asan $(LDLIBS)
	python3 test/tiny_olmoe.py write $(GOLDEN_GGUF)
	python3 test/fuzz_mutate.py build/engine-asan $(GOLDEN_GGUF) --iterations $(or $(ITER),600)
	python3 test/fuzz_mutate.py build/engine-asan $(FIXTURE) --iterations $(or $(ITER),600) --seed 7

# Address + undefined-behaviour sanitizers on the unit tests and a threaded streaming run, then the thread sanitizer on the same run.
SAN_RUN = stream $(SMALL) --tokens 60 --cache-mb 1 --stage 1 --io-threads 3 --smart 1,8,1 --policy adaptive
sanitize: build
	$(CC) -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc $(SRC) test/test_engine.c -o build/test-asan $(LDLIBS)
	./build/test-asan $(FIXTURE)
	$(CC) -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer $(SRC) src/main.c -o build/engine-asan $(LDLIBS)
	$(CC) -O1 -g -std=c11 -fsanitize=thread $(SRC) src/main.c -o build/engine-tsan $(LDLIBS)
	@if [ -n "$(SMALL)" ]; then ./build/engine-asan $(SAN_RUN) >/dev/null && ./build/engine-tsan $(SAN_RUN) >/dev/null && echo "sanitizers clean on $(SMALL)"; else echo "(set SMALL=<moe.gguf> to also run the streaming engine under the sanitizers)"; fi
# Compares tokenization, logits and routing with llama.cpp (needs llama-server and llama-tokenize on PATH): make verify-llama MODEL=path/to/olmoe.gguf
verify-llama: $(BIN)
	python3 test/verify_llama.py $(MODEL) --engine $(BIN)
clean:
	rm -rf build
.PHONY: fuzz golden golden-files golden-paths verify-llama all test test-paths test-avx2 sanitize clean
