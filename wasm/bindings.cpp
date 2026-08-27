// WebAssembly entry points.
//
// Only the search path is compiled to wasm. Building an index is a one-time,
// minutes-long job that happens offline with the CLI; shipping the build code
// into the browser would roughly double the module for something no visitor
// ever runs.
//
// The index arrives as a byte buffer the page fetched, rather than through the
// mmap path -- there is no mmap in a browser. GraphView is what makes that a
// non-event: it is just pointers, so pointing it at a heap buffer instead of a
// mapping needs no second implementation of the search.
#include "vec/format.hpp"
#include "vec/hnsw.hpp"

#include <cstdint>
#include <cstring>
#include <emscripten/emscripten.h>

#include <chrono>
#include <vector>

namespace {

std::vector<std::byte> g_index;   // the fetched index file, owned here
vec::GraphView g_view;
vec::Scratch g_scratch;
bool g_ready = false;

// Results are handed back through a flat buffer the JS side reads directly,
// which avoids marshalling a structure across the boundary per query.
std::vector<std::uint32_t> g_ids;
std::vector<float> g_dists;
double g_last_us = 0;

}  // namespace

extern "C" {

// Loads an index from bytes already in the wasm heap. Returns 1 on success.
EMSCRIPTEN_KEEPALIVE
int hnsw_load(const std::uint8_t* data, int length) {
    try {
        g_index.assign(reinterpret_cast<const std::byte*>(data),
                       reinterpret_cast<const std::byte*>(data) + length);

        vec::IndexHeader h{};
        if (g_index.size() < sizeof(h)) return 0;
        std::memcpy(&h, g_index.data(), sizeof(h));
        if (h.magic != vec::kMagic || h.version != vec::kVersion) return 0;

        const std::byte* base = g_index.data();
        auto in_bounds = [&](std::uint64_t off, std::uint64_t len) {
            return off + len <= g_index.size();
        };
        if (!in_bounds(h.vectors_off, h.vectors_len) ||
            !in_bounds(h.level0_off, h.level0_len) ||
            !in_bounds(h.upper_off_off, h.upper_off_len) ||
            !in_bounds(h.upper_off, h.upper_len) ||
            !in_bounds(h.levels_off, h.levels_len)) {
            return 0;
        }

        g_view.vectors = reinterpret_cast<const float*>(base + h.vectors_off);
        g_view.dim = h.dim;
        g_view.stride = h.stride;
        g_view.count = h.count;
        g_view.level0 = reinterpret_cast<const std::uint32_t*>(base + h.level0_off);
        g_view.upper_off =
            reinterpret_cast<const std::uint64_t*>(base + h.upper_off_off);
        g_view.upper = h.upper_len ? reinterpret_cast<const std::uint32_t*>(
                                         base + h.upper_off)
                                   : nullptr;
        g_view.levels = reinterpret_cast<const std::int32_t*>(base + h.levels_off);
        g_view.max_m0 = h.max_m0;
        g_view.M = h.M;
        g_view.entry = h.entry;
        g_view.max_level = h.max_level;
        g_view.metric = h.metric == 0 ? vec::Metric::kL2 : vec::Metric::kInnerProduct;

        g_ready = true;
        return 1;
    } catch (...) {
        g_ready = false;
        return 0;
    }
}

EMSCRIPTEN_KEEPALIVE int hnsw_count() { return g_ready ? (int)g_view.count : 0; }
EMSCRIPTEN_KEEPALIVE int hnsw_dim() { return g_ready ? (int)g_view.dim : 0; }

// Graph search. Returns the number of results, readable via hnsw_ids/hnsw_dists.
EMSCRIPTEN_KEEPALIVE
int hnsw_search(const float* query, int k, int ef) {
    if (!g_ready) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto r = vec::search(g_view, query, k, ef, g_scratch);
    g_last_us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0).count();

    g_ids.clear();
    g_dists.clear();
    for (const auto& n : r) { g_ids.push_back(n.id); g_dists.push_back(n.dist); }
    return static_cast<int>(r.size());
}

// Exhaustive search over the same vectors, so the page can show the speedup
// rather than assert it.
EMSCRIPTEN_KEEPALIVE
int hnsw_search_bruteforce(const float* query, int k) {
    if (!g_ready) return 0;
    const auto t0 = std::chrono::steady_clock::now();

    std::vector<vec::Neighbor> best;
    best.reserve(g_view.count);
    for (std::uint32_t i = 0; i < g_view.count; ++i)
        best.push_back({g_view.dist_to(query, i), i});
    const std::size_t kk = std::min<std::size_t>(k, best.size());
    std::partial_sort(best.begin(), best.begin() + kk, best.end());
    best.resize(kk);

    g_last_us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0).count();

    g_ids.clear();
    g_dists.clear();
    for (const auto& n : best) { g_ids.push_back(n.id); g_dists.push_back(n.dist); }
    return static_cast<int>(best.size());
}

EMSCRIPTEN_KEEPALIVE const std::uint32_t* hnsw_ids() { return g_ids.data(); }
EMSCRIPTEN_KEEPALIVE const float* hnsw_dists() { return g_dists.data(); }
EMSCRIPTEN_KEEPALIVE double hnsw_last_us() { return g_last_us; }

}  // extern "C"
