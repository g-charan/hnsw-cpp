#pragma once

#include "core/arena.hpp"
#include "vec/distance.hpp"
#include "vec/flat.hpp"
#include "vec/store.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace vec {

// Hierarchical Navigable Small World index (Malkov & Yashunin, 2016).
//
// Layout notes, which are most of what separates this from a direct
// transcription of the paper:
//
//   * Layer 0 is one flat uint32 array of count*(max_m0+1) entries -- a degree
//     followed by its neighbour ids, inline. Layer 0 holds every node and takes
//     essentially all the search traffic, so removing the per-node indirection
//     there is the allocation decision that matters.
//   * Upper layers are sparse (a node reaches level l with probability about
//     1/M^l), so those lists are packed rather than stored as a dense array
//     that would be almost entirely empty.
//   * The visited set is generation-stamped, never cleared. Clearing a
//     count-entry set per query would come to dominate the query itself.

// ---------------------------------------------------------------- view ------

// A read-only handle onto a graph: raw pointers, no ownership, no allocation.
//
// This exists so one search implementation serves both a freshly built index
// and one mapped straight off disk. Without it the mapped path would need its
// own copy of the beam search and the two would drift apart.
struct GraphView {
    const float* vectors = nullptr;
    std::size_t dim = 0, stride = 0, count = 0;

    const std::uint32_t* level0 = nullptr;    // count*(max_m0+1), inline degrees
    const std::uint32_t* upper = nullptr;     // packed upper-layer lists
    const std::uint64_t* upper_off = nullptr; // per node, index into `upper`
    const std::int32_t* levels = nullptr;

    std::size_t max_m0 = 0, M = 0;
    std::uint32_t entry = 0;
    std::int32_t max_level = 0;
    Metric metric = Metric::kL2;

    const float* vec(std::uint32_t id) const {
        return vectors + static_cast<std::size_t>(id) * stride;
    }
    const std::uint32_t* links(std::uint32_t id, int layer) const {
        if (layer == 0) return level0 + static_cast<std::size_t>(id) * (max_m0 + 1);
        return upper + upper_off[id] + static_cast<std::size_t>(layer - 1) * (M + 1);
    }
    float dist_to(const float* q, std::uint32_t id) const {
        return distance(metric, q, vec(id), dim);
    }
};

// Per-thread search scratch. Searching never writes to the graph, so any number
// of threads may share one index, but each needs its own visited set and heaps.
// Handing that out explicitly is what keeps allocation off the query path.
struct Scratch {
    std::vector<std::uint32_t> visited;
    std::uint32_t gen = 0;
    std::vector<Neighbor> candidates;  // min-heap on distance
    std::vector<Neighbor> results;     // max-heap on distance

    void reset_for(std::size_t n) {
        if (visited.size() < n) { visited.assign(n, 0); gen = 0; }
        // Wrapping would alias stale marks as visited, so on wrap the set is
        // cleared once and the counter restarts.
        if (++gen == 0) { std::fill(visited.begin(), visited.end(), 0); gen = 1; }
        candidates.clear();
        results.clear();
    }
};

// -------------------------------------------------------------- search ------

// Beam of one: follow the locally best edge until no neighbour improves. Used
// for layers above the target, which exist only to reach the right region cheaply.
inline std::uint32_t greedy_descend(const GraphView& g, const float* q,
                                    std::uint32_t ep, int layer) {
    float best = g.dist_to(q, ep);
    bool moved = true;
    while (moved) {
        moved = false;
        const std::uint32_t* l = g.links(ep, layer);
        const std::uint32_t d = l[0];
        for (std::uint32_t i = 0; i < d; ++i) {
            const std::uint32_t cand = l[1 + i];
            const float dc = g.dist_to(q, cand);
            if (dc < best) { best = dc; ep = cand; moved = true; }
        }
    }
    return ep;
}

// Best-first search bounded to a beam of `ef`. Leaves the beam in s.results as
// a max-heap on distance.
inline void search_layer(const GraphView& g, const float* q, std::uint32_t ep,
                         std::size_t ef, int layer, Scratch& s) {
    const float d0 = g.dist_to(q, ep);
    s.visited[ep] = s.gen;
    s.candidates.clear();
    s.results.clear();
    s.candidates.push_back(Neighbor{d0, ep});
    s.results.push_back(Neighbor{d0, ep});

    while (!s.candidates.empty()) {
        std::pop_heap(s.candidates.begin(), s.candidates.end(),
                      std::greater<Neighbor>{});
        const Neighbor c = s.candidates.back();
        s.candidates.pop_back();

        // Everything still queued is further than the worst result being kept,
        // so no further expansion can improve the beam.
        if (s.results.size() >= ef && c.dist > s.results.front().dist) break;

        const std::uint32_t* l = g.links(c.id, layer);
        const std::uint32_t deg = l[0];
        for (std::uint32_t i = 0; i < deg; ++i) {
            const std::uint32_t e = l[1 + i];
            if (s.visited[e] == s.gen) continue;
            s.visited[e] = s.gen;

            const float de = g.dist_to(q, e);
            if (s.results.size() < ef || de < s.results.front().dist) {
                s.candidates.push_back(Neighbor{de, e});
                std::push_heap(s.candidates.begin(), s.candidates.end(),
                               std::greater<Neighbor>{});
                s.results.push_back(Neighbor{de, e});
                std::push_heap(s.results.begin(), s.results.end());
                if (s.results.size() > ef) {
                    std::pop_heap(s.results.begin(), s.results.end());
                    s.results.pop_back();
                }
            }
        }
    }
}

// Top-k nearest, nearest first. `ef` is the beam width, and the dial that
// trades query time against recall.
inline std::vector<Neighbor> search(const GraphView& g, const float* q,
                                    std::size_t k, std::size_t ef, Scratch& s) {
    if (g.count == 0 || k == 0) return {};
    ef = std::max(ef, k);

    std::uint32_t ep = g.entry;
    for (int l = g.max_level; l > 0; --l) ep = greedy_descend(g, q, ep, l);

    s.reset_for(g.count);
    search_layer(g, q, ep, ef, 0, s);

    std::vector<Neighbor> out = s.results;
    std::sort_heap(out.begin(), out.end());  // ascending by distance
    if (out.size() > k) out.resize(k);
    return out;
}

// Fraction of the k nearest that the beam recovered, against a graph's own
// exhaustive answer. Kept next to search so the two never disagree on ordering.
inline double graph_recall(const GraphView& g, const float* q, std::size_t k,
                           std::size_t ef, const std::uint32_t* truth, Scratch& s) {
    return recall_at_k(search(g, q, k, ef, s), truth, k);
}

// ------------------------------------------------------------- builder ------

class HnswIndex {
public:
    struct Params {
        std::size_t M = 16;                 // edges per node on upper layers
        std::size_t ef_construction = 200;  // beam width while building
        std::uint64_t seed = 100;
    };

    // Two constructors rather than a defaulted Params argument: a default
    // argument of Params{} cannot be written inside the class that defines
    // Params, because its member initialisers are not yet complete there.
    HnswIndex(std::size_t dim, std::size_t capacity, Metric metric = Metric::kL2)
        : HnswIndex(dim, capacity, metric, Params{}) {}

    HnswIndex(std::size_t dim, std::size_t capacity, Metric metric, Params params)
        : store_(dim, capacity), metric_(metric), params_(params),
          capacity_(capacity), max_m0_(params.M * 2), rng_(params.seed),
          level_mult_(1.0 / std::log(static_cast<double>(params.M))) {
        if (params.M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");

        levels_.reserve(capacity);
        upper_off_.reserve(capacity);
        // Layer 0 is dense and taken up front: one contiguous block for the
        // whole graph, so neighbour lists of nearby ids share cache lines.
        level0_ = arena_.alloc_array<std::uint32_t>(capacity * (max_m0_ + 1), 64);
        std::memset(level0_, 0, capacity * (max_m0_ + 1) * sizeof(std::uint32_t));
    }

    std::size_t dim() const { return store_.dim(); }
    std::size_t size() const { return store_.size(); }
    int max_level() const { return max_level_; }
    const VectorStore& store() const { return store_; }
    const Params& params() const { return params_; }
    Metric metric() const { return metric_; }
    std::uint32_t entry_point() const { return entry_; }
    const std::vector<std::int32_t>& levels() const { return levels_; }
    const std::vector<std::uint64_t>& upper_offsets() const { return upper_off_; }
    const std::uint32_t* level0_data() const { return level0_; }
    const std::vector<std::uint32_t>& upper_data() const { return upper_; }
    std::size_t max_m0() const { return max_m0_; }

    GraphView view() const {
        GraphView g;
        g.vectors = store_.get(0);
        g.dim = store_.dim();
        g.stride = store_.stride();
        g.count = store_.size();
        g.level0 = level0_;
        g.upper = upper_.empty() ? nullptr : upper_.data();
        g.upper_off = upper_off_.empty() ? nullptr : upper_off_.data();
        g.levels = levels_.empty() ? nullptr : levels_.data();
        g.max_m0 = max_m0_;
        g.M = params_.M;
        g.entry = entry_;
        g.max_level = max_level_;
        g.metric = metric_;
        return g;
    }

    std::uint32_t add(const float* v) {
        if (store_.size() >= capacity_) throw std::runtime_error("HnswIndex full");

        const auto id = static_cast<std::uint32_t>(store_.add(v));
        const int level = sample_level();
        levels_.push_back(level);

        // Upper lists are packed end to end; each node records where its own
        // run starts. Storing the offset rather than a pointer is what lets the
        // same layout be mapped straight off disk with no fix-up pass.
        upper_off_.push_back(upper_.size());
        if (level > 0) {
            upper_.resize(upper_.size() +
                          static_cast<std::size_t>(level) * (params_.M + 1), 0);
        }

        if (id == 0) { entry_ = 0; max_level_ = level; return id; }

        const GraphView g = view();
        const float* q = store_.get(id);
        std::uint32_t ep = entry_;

        for (int l = max_level_; l > level; --l) ep = greedy_descend(g, q, ep, l);

        for (int l = std::min(level, max_level_); l >= 0; --l) {
            build_scratch_.reset_for(store_.size());
            search_layer(g, q, ep, params_.ef_construction, l, build_scratch_);

            const std::size_t m = (l == 0) ? max_m0_ : params_.M;
            auto chosen = select_neighbors(build_scratch_.results, m);
            for (const Neighbor& n : chosen) connect(id, n.id, l);
            if (!chosen.empty()) ep = chosen.front().id;
        }

        if (level > max_level_) { max_level_ = level; entry_ = id; }
        return id;
    }

    void add_many(const float* vs, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) add(vs + i * store_.dim());
    }

    std::vector<Neighbor> search(const float* q, std::size_t k, std::size_t ef,
                                 Scratch& s) const {
        return vec::search(view(), q, k, ef, s);
    }

    // Convenience overload. Not safe from several threads at once -- pass a
    // Scratch for that.
    std::vector<Neighbor> search(const float* q, std::size_t k,
                                 std::size_t ef) const {
        return vec::search(view(), q, k, ef, query_scratch_);
    }

    std::size_t graph_bytes() const {
        return arena_.bytes_reserved() + upper_.size() * sizeof(std::uint32_t);
    }

    double average_degree() const {
        if (store_.size() == 0) return 0.0;
        std::size_t total = 0;
        for (std::size_t i = 0; i < store_.size(); ++i)
            total += level0_[i * (max_m0_ + 1)];
        return static_cast<double>(total) / static_cast<double>(store_.size());
    }

private:
    std::uint32_t* links_mut(std::uint32_t id, int layer) {
        if (layer == 0) return level0_ + static_cast<std::size_t>(id) * (max_m0_ + 1);
        return upper_.data() + upper_off_[id] +
               static_cast<std::size_t>(layer - 1) * (params_.M + 1);
    }

    int sample_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng_);
        if (r <= 0.0) r = std::numeric_limits<double>::min();
        return static_cast<int>(-std::log(r) * level_mult_);
    }

    float dist_to(const float* q, std::uint32_t id) const {
        return distance(metric_, q, store_.get(id), store_.dim());
    }

    // Algorithm 4 from the paper. Taking the m nearest candidates outright
    // clusters every edge in whichever direction the data is densest, and the
    // graph stops being navigable from the sparse side. Keeping a candidate
    // only when it is closer to the query than to any already-chosen neighbour
    // spreads edges across directions, and is worth several points of recall at
    // the same degree.
    std::vector<Neighbor> select_neighbors(const std::vector<Neighbor>& cands,
                                           std::size_t m) const {
        std::vector<Neighbor> pool = cands;
        std::sort(pool.begin(), pool.end());  // nearest first

        std::vector<Neighbor> chosen;
        chosen.reserve(m);
        for (const Neighbor& c : pool) {
            if (chosen.size() >= m) break;
            bool keep = true;
            for (const Neighbor& r : chosen) {
                const float d_cr = distance(metric_, store_.get(c.id),
                                            store_.get(r.id), store_.dim());
                if (d_cr < c.dist) { keep = false; break; }
            }
            if (keep) chosen.push_back(c);
        }
        return chosen;
    }

    void connect(std::uint32_t a, std::uint32_t b, int layer) {
        if (a == b) return;
        const std::size_t cap = (layer == 0) ? max_m0_ : params_.M;
        add_link(a, b, layer, cap);
        add_link(b, a, layer, cap);
    }

    void add_link(std::uint32_t from, std::uint32_t to, int layer, std::size_t cap) {
        std::uint32_t* l = links_mut(from, layer);
        const std::uint32_t d = l[0];
        for (std::uint32_t i = 0; i < d; ++i) {
            if (l[1 + i] == to) return;  // already linked
        }
        if (d < cap) { l[1 + d] = to; l[0] = d + 1; return; }

        // Full: re-run the selection heuristic over the existing neighbours plus
        // the newcomer, from `from`'s point of view. Simply dropping the
        // furthest would degrade the graph the same way naive top-m does.
        const float* base = store_.get(from);
        std::vector<Neighbor> cands;
        cands.reserve(d + 1);
        for (std::uint32_t i = 0; i < d; ++i)
            cands.push_back(Neighbor{dist_to(base, l[1 + i]), l[1 + i]});
        cands.push_back(Neighbor{dist_to(base, to), to});

        auto kept = select_neighbors(cands, cap);
        l[0] = static_cast<std::uint32_t>(kept.size());
        for (std::size_t i = 0; i < kept.size(); ++i) l[1 + i] = kept[i].id;
    }

    core::Arena arena_{1u << 22};
    VectorStore store_;
    Metric metric_;
    Params params_;
    std::size_t capacity_;
    std::size_t max_m0_;

    std::uint32_t* level0_ = nullptr;
    std::vector<std::uint32_t> upper_;
    std::vector<std::uint64_t> upper_off_;
    std::vector<std::int32_t> levels_;

    std::uint32_t entry_ = 0;
    int max_level_ = 0;

    std::mt19937_64 rng_;
    double level_mult_;

    Scratch build_scratch_;
    mutable Scratch query_scratch_;
};

}  // namespace vec
