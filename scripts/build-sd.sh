#!/usr/bin/env bash
# Builds stable-diffusion.cpp (MIT, https://github.com/leejet/stable-diffusion.cpp) as static libraries: the image engine that Clutor's desktop app links in (app/src-tauri/build.rs).
# Usage: scripts/build-sd.sh [work dir]   (default ~/.cache/clutor-sd). Prints the source/build directory, which is what build.rs reads as CLUTOR_SD_DIR.
# macOS (Apple silicon) uses the Apple GPU through Metal. Linux and Windows use the processor (x86-64 with AVX2); a GPU backend for them is not built here.
set -euo pipefail
SD_REV="fc350be2ceeeb859bcc201ac435591ed7d46821d"   # the revision the image engine was tested with
DIR="${1:-$HOME/.cache/clutor-sd}"
SRC="$DIR/src"
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  mkdir -p "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" remote add origin https://github.com/leejet/stable-diffusion.cpp
  git -C "$SRC" fetch -q --depth 1 origin "$SD_REV"
  git -C "$SRC" checkout -q FETCH_HEAD
  git -C "$SRC" submodule update --init --recursive --depth 1 -q
fi
FLAGS=(-DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON)   # position independent: the app is a shared library on Linux
case "$(uname -s)" in
  Darwin) [ "$(uname -m)" = "arm64" ] && FLAGS+=(-DSD_METAL=ON) ;;
  *) FLAGS+=(-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON) ;;   # a program that runs on any x86-64 processor from the last ten years, not only the build machine's
esac
cmake -S "$SRC" -B "$SRC/build" "${FLAGS[@]}" >&2
# Linux runners have 4 cores and 16 GB: an unbounded -j once starved the hosted runner until it lost contact with the server, so the parallelism is capped there.
JOBS=()
[ "$(uname -s)" = "Linux" ] && JOBS=(-j 2) || JOBS=(-j)
cmake --build "$SRC/build" --config Release --target stable-diffusion "${JOBS[@]}" >&2
echo "$SRC"
if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "dir=$SRC" >> "$GITHUB_OUTPUT"; fi
