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
// Layout notes, which are most of what makes this faster than a textbook
// transcription of the paper:
//
//   * Layer 0 is one flat uint32 array of n * (2M+1) entries -- a degree
//     followed by its neighbour ids, inline. No per-node vector, no pointer
//     chase to reach a neighbour list. Layer 0 holds every node and takes
//     essentially all the search traffic, so this is the allocation that
//     matters.
//   * Upper layers are sparse (a node reaches level l with probability
//     ~1/M^l), so those lists are arena-allocated per node instead of a dense
//     array that would be almost entirely empty.
//   * The visited set is a generation-stamped array, never cleared. Clearing an
//     n-bit set per query would dominate the query itself once n is large.
class HnswIndex {
public:
    struct Params {
        std::size_t M = 16;                // edges per node on upper layers
        std::size_t ef_construction = 200; // beam width while building
        std::uint64_t seed = 100;
    };

    // Per-thread scratch. Searching is read-only against the graph, so many
    // threads can share an index, but each needs its own visited set and heaps.
    // Handing that out explicitly keeps every allocation off the query path.
    struct Scratch {
        std::vector<std::uint32_t> visited;
        std::uint32_t gen = 0;
        std::vector<Neighbor> candidates;  // min-heap on distance
        std::vector<Neighbor> results;     // max-heap on distance
        std::vector<Neighbor> work;

        void reset_for(std::size_t n) {
            if (visited.size() < n) {
                visited.assign(n, 0);
                gen = 0;
            }
            // Wrapping the generation counter would alias stale marks as
            // visited, so on wrap the set is cleared once and restarted.
            if (++gen == 0) {
                std::fill(visited.begin(), visited.end(), 0);
                gen = 1;
            }
            candidates.clear();
            results.clear();
        }
    };

    // Two constructors rather than a defaulted Params argument: a default
    // argument of Params{} cannot be written inside the class that defines
    // Params, because its member initialisers are not complete yet.
    HnswIndex(std::size_t dim, std::size_t capacity, Metric metric = Metric::kL2)
        : HnswIndex(dim, capacity, metric, Params{}) {}

    HnswIndex(std::size_t dim, std::size_t capacity, Metric metric, Params params)
        : store_(dim, capacity), metric_(metric), params_(params),
          capacity_(capacity), max_m0_(params.M * 2), rng_(params.seed),
          level_mult_(1.0 / std::log(static_cast<double>(params.M))) {
        if (params.M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");

        levels_.reserve(capacity);
        upper_.reserve(capacity);
        // Layer 0 is dense and allocated up front: one contiguous block for the
        // entire graph, so neighbour lists of nearby ids share cache lines.
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

    // ------------------------------------------------------------- build ----

    std::uint32_t add(const float* v) {
        if (store_.size() >= capacity_) throw std::runtime_error("HnswIndex full");

        const auto id = static_cast<std::uint32_t>(store_.add(v));
        const int level = sample_level();
        levels_.push_back(level);
        upper_.push_back(nullptr);

        if (level > 0) {
            // level slots, each a degree plus M neighbour ids.
            auto* mem = arena_.alloc_array<std::uint32_t>(
                static_cast<std::size_t>(level) * (params_.M + 1), 64);
            std::memset(mem, 0,
                        static_cast<std::size_t>(level) * (params_.M + 1) *
                            sizeof(std::uint32_t));
            upper_[id] = mem;
        }

        if (id == 0) {
            entry_ = 0;
            max_level_ = level;
            return id;
        }

        const float* q = store_.get(id);
        std::uint32_t ep = entry_;

        // Descend the layers above this node's own with a beam of one: these
        // levels exist only to get close to the right region cheaply.
        for (int l = max_level_; l > level; --l) {
            ep = greedy_descend(q, ep, l);
        }

        // Then insert into every layer this node belongs to.
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            build_scratch_.reset_for(store_.size());
            search_layer(q, ep, params_.ef_construction, l, build_scratch_);

            const std::size_t m = (l == 0) ? max_m0_ : params_.M;
            auto chosen = select_neighbors(build_scratch_.results, m);

            for (const Neighbor& n : chosen) connect(id, n.id, l);
            if (!chosen.empty()) ep = chosen.front().id;
        }

        if (level > max_level_) {
            max_level_ = level;
            entry_ = id;
        }
        return id;
    }

    void add_many(const float* vs, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) add(vs + i * store_.dim());
    }

    // ------------------------------------------------------------ search ----

    std::vector<Neighbor> search(const float* q, std::size_t k, std::size_t ef,
                                 Scratch& s) const {
        if (store_.size() == 0) return {};
        ef = std::max(ef, k);

        std::uint32_t ep = entry_;
        for (int l = max_level_; l > 0; --l) ep = greedy_descend(q, ep, l);

        s.reset_for(store_.size());
        search_layer(q, ep, ef, 0, s);

        // results is a max-heap of up to ef; take the k smallest, nearest first.
        std::vector<Neighbor> out = s.results;
        std::sort_heap(out.begin(), out.end());  // ascending by distance
        if (out.size() > k) out.resize(k);
        return out;
    }

    // Convenience overload. Not safe to call from several threads at once --
    // use the Scratch overload for that.
    std::vector<Neighbor> search(const float* q, std::size_t k,
                                 std::size_t ef) const {
        return search(q, k, ef, query_scratch_);
    }

    // ------------------------------------------------------------- stats ----

    std::size_t graph_bytes() const { return arena_.bytes_reserved(); }

    double average_degree() const {
        if (store_.size() == 0) return 0.0;
        std::size_t total = 0;
        for (std::size_t i = 0; i < store_.size(); ++i) total += level0_[i * (max_m0_ + 1)];
        return static_cast<double>(total) / static_cast<double>(store_.size());
    }

private:
    // ------------------------------------------------------ neighbour access

    std::uint32_t* links(std::uint32_t id, int layer) const {
        if (layer == 0) return level0_ + static_cast<std::size_t>(id) * (max_m0_ + 1);
        return upper_[id] + static_cast<std::size_t>(layer - 1) * (params_.M + 1);
    }
    static std::uint32_t degree(const std::uint32_t* l) { return l[0]; }
    static std::uint32_t* neighbors(std::uint32_t* l) { return l + 1; }

    int sample_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng_);
        if (r <= 0.0) r = std::numeric_limits<double>::min();
        return static_cast<int>(-std::log(r) * level_mult_);
    }

    float dist_to(const float* q, std::uint32_t id) const {
        return distance(metric_, q, store_.get(id), store_.dim());
    }

    // ------------------------------------------------------------- search ---

    // Beam of one, used for the layers above the target: follow the locally
    // best edge until no neighbour improves on the current node.
    std::uint32_t greedy_descend(const float* q, std::uint32_t ep, int layer) const {
        float best = dist_to(q, ep);
        bool moved = true;
        while (moved) {
            moved = false;
            const std::uint32_t* l = links(ep, layer);
            const std::uint32_t d = degree(l);
            for (std::uint32_t i = 0; i < d; ++i) {
                const std::uint32_t cand = l[1 + i];
                const float dc = dist_to(q, cand);
                if (dc < best) {
                    best = dc;
                    ep = cand;
                    moved = true;
                }
            }
        }
        return ep;
    }

    // Best-first search bounded to a beam of `ef`. Leaves the result set in
    // s.results as a max-heap on distance.
    void search_layer(const float* q, std::uint32_t ep, std::size_t ef, int layer,
                      Scratch& s) const {
        const float d0 = dist_to(q, ep);
        s.visited[ep] = s.gen;

        s.candidates.clear();
        s.results.clear();
        s.candidates.push_back(Neighbor{d0, ep});
        s.results.push_back(Neighbor{d0, ep});

        while (!s.candidates.empty()) {
            // Closest unexpanded candidate.
            std::pop_heap(s.candidates.begin(), s.candidates.end(),
                          std::greater<Neighbor>{});
            const Neighbor c = s.candidates.back();
            s.candidates.pop_back();

            // Everything left in the queue is further than the worst result we
            // are keeping, so no expansion can improve the beam.
            if (s.results.size() >= ef && c.dist > s.results.front().dist) break;

            const std::uint32_t* l = links(c.id, layer);
            const std::uint32_t deg = degree(l);
            for (std::uint32_t i = 0; i < deg; ++i) {
                const std::uint32_t e = l[1 + i];
                if (s.visited[e] == s.gen) continue;
                s.visited[e] = s.gen;

                const float de = dist_to(q, e);
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

    // ---------------------------------------------------------- neighbour ---

    // Algorithm 4 from the paper. Taking the M nearest candidates outright
    // clusters every edge in whichever direction the data is densest, and the
    // graph stops being navigable from the sparse side. This keeps a candidate
    // only when it is closer to the query than to any neighbour already chosen,
    // which spreads edges across directions and is worth several points of
    // recall at the same degree.
    std::vector<Neighbor> select_neighbors(const std::vector<Neighbor>& cands,
                                           std::size_t m) const {
        std::vector<Neighbor> pool = cands;
        std::sort(pool.begin(), pool.end());  // nearest first

        std::vector<Neighbor> chosen;
        chosen.reserve(m);
        for (const Neighbor& c : pool) {
            if (chosen.size() >= m) break;
            if (c.id == kNoId) continue;

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

    // Add a bidirectional edge, pruning the far end if it overflows.
    void connect(std::uint32_t a, std::uint32_t b, int layer) {
        if (a == b) return;
        const std::size_t cap = (layer == 0) ? max_m0_ : params_.M;
        add_link(a, b, layer, cap);
        add_link(b, a, layer, cap);
    }

    void add_link(std::uint32_t from, std::uint32_t to, int layer, std::size_t cap) {
        std::uint32_t* l = links(from, layer);
        const std::uint32_t d = degree(l);

        for (std::uint32_t i = 0; i < d; ++i) {
            if (l[1 + i] == to) return;  // already linked
        }

        if (d < cap) {
            l[1 + d] = to;
            l[0] = d + 1;
            return;
        }

        // Full. Re-run the selection heuristic over the existing neighbours
        // plus the newcomer, from `from`'s point of view, and keep the best cap
        // of them. Dropping the furthest instead would degrade the graph the
        // same way naive top-M selection does.
        const float* base = store_.get(from);
        std::vector<Neighbor> cands;
        cands.reserve(d + 1);
        for (std::uint32_t i = 0; i < d; ++i) {
            cands.push_back(Neighbor{dist_to(base, l[1 + i]), l[1 + i]});
        }
        cands.push_back(Neighbor{dist_to(base, to), to});

        auto kept = select_neighbors(cands, cap);
        l[0] = static_cast<std::uint32_t>(kept.size());
        for (std::size_t i = 0; i < kept.size(); ++i) l[1 + i] = kept[i].id;
    }

    static constexpr std::uint32_t kNoId = 0xFFFFFFFFu;

    core::Arena arena_{1u << 22};
    VectorStore store_;
    Metric metric_;
    Params params_;
    std::size_t capacity_;
    std::size_t max_m0_;

    std::uint32_t* level0_ = nullptr;
    std::vector<std::uint32_t*> upper_;
    std::vector<int> levels_;

    std::uint32_t entry_ = 0;
    int max_level_ = 0;

    std::mt19937_64 rng_;
    double level_mult_;

    Scratch build_scratch_;
    mutable Scratch query_scratch_;
};

}  // namespace vec
