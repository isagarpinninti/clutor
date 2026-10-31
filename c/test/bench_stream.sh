#!/bin/sh
# Run on a model file that nothing has mapped or read without F_NOCACHE since it was written (copy it with fastcopy), or the streamed runs read RAM.
# Real forward pass, same prompt and tokens, mapped file against streamed experts at several cache sizes. Usage: bench_stream.sh MODEL.gguf [TOKENS]
M=$1; N=${2:-40}; E=$(dirname "$0")/../build/clutor-engine
P="Once upon a time, in a small village by the sea, there lived an old fisherman who"
run() { label=$1; shift; out=$("$E" generate "$M" "$P" --n "$N" "$@" 2>/tmp/bench.err); echo "$label|$(printf '%s' "$out" | md5)|$(grep -o '[0-9.]* s/token' /tmp/bench.err | head -1)|$(grep -o 'streaming:.*' /tmp/bench.err | sed 's/ of [0-9]* bytes//')"; }
for mb in 128 256 512 1024; do
  run "stream ${mb}MiB adaptive+prefetch" --cache-mb $mb --no-os-cache
done
run "stream 256MiB lru+prefetch" --cache-mb 256 --policy lru --no-os-cache
run "stream 256MiB adaptive no-prefetch" --cache-mb 256 --prefetch off --no-os-cache
# The mapped run goes LAST: it pulls the whole file into the OS cache, which would make every later streamed run read from RAM, not the disk.
run "mapped (whole file resident)"
