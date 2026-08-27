#pragma once

#include "vec/distance.hpp"
#include "vec/store.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

namespace vec {

struct Neighbor {
    float dist;
    std::uint32_t id;
    // Ordered by distance so a std::priority_queue of these is a max-heap on
    // distance, which is what bounded top-k selection wants.
    bool operator<(const Neighbor& o) const { return dist < o.dist; }
    bool operator>(const Neighbor& o) const { return dist > o.dist; }
};

// Exhaustive search. This is the ground truth the graph index is scored
// against, and the baseline its speedup is quoted over, so it stays simple
// enough to be obviously correct.
class FlatIndex {
public:
    FlatIndex(std::size_t dim, std::size_t capacity, Metric metric = Metric::kL2)
        : store_(dim, capacity), metric_(metric) {}

    std::size_t add(const float* v) { return store_.add(v); }
    void add_many(const float* vs, std::size_t n) { store_.add_many(vs, n); }

    std::vector<Neighbor> search(const float* q, std::size_t k) const {
        // Bounded max-heap: keep the k best seen, evict the worst. O(n log k)
        // with k tiny, rather than sorting all n.
        std::priority_queue<Neighbor> heap;
        const std::size_t n = store_.size();
        for (std::size_t i = 0; i < n; ++i) {
            const float d = distance(metric_, q, store_.get(i), store_.dim());
            if (heap.size() < k) {
                heap.push(Neighbor{d, static_cast<std::uint32_t>(i)});
            } else if (d < heap.top().dist) {
                heap.pop();
                heap.push(Neighbor{d, static_cast<std::uint32_t>(i)});
            }
        }

        std::vector<Neighbor> out(heap.size());
        for (std::size_t i = heap.size(); i-- > 0;) {  // heap pops worst-first
            out[i] = heap.top();
            heap.pop();
        }
        return out;
    }

    const VectorStore& store() const { return store_; }
    std::size_t size() const { return store_.size(); }
    std::size_t dim() const { return store_.dim(); }
    Metric metric() const { return metric_; }

private:
    VectorStore store_;
    Metric metric_;
};

// Fraction of the true k nearest that `got` actually found. This is the number
// the whole project is judged on, so it is defined once, here, and every recall
// figure in the README comes through it.
inline double recall_at_k(const std::vector<Neighbor>& got,
                          const std::uint32_t* truth, std::size_t k) {
    if (k == 0) return 1.0;
    std::size_t hits = 0;
    for (std::size_t i = 0; i < got.size() && i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            if (got[i].id == truth[j]) { ++hits; break; }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

}  // namespace vec
