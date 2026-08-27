#!/usr/bin/env bash
# Builds the search path to WebAssembly. Requires emscripten (brew install emscripten).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/web/public"
mkdir -p "$out"

emcc "$root/wasm/bindings.cpp" \
  -I"$root/include" \
  -O3 -std=c++23 \
  -msimd128 \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=createHnswModule \
  -s EXPORTED_FUNCTIONS='["_hnsw_load","_hnsw_search","_hnsw_search_bruteforce","_hnsw_ids","_hnsw_dists","_hnsw_count","_hnsw_dim","_hnsw_last_us","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPF32","HEAPU32","getValue"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=64MB \
  -o "$out/hnsw.js"

echo "wrote $out/hnsw.js and $out/hnsw.wasm"
