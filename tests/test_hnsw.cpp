// Correctness for an approximate index means recall against exhaustive search,
// so every check here scores the graph with FlatIndex as the oracle.
#include "check.hpp"
#include "vec/flat.hpp"
#include "vec/hnsw.hpp"

#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Dataset {
    std::size_t dim, n;
    std::vector<float> data;
    std::vector<float> queries;
    std::size_t nq;

    Dataset(std::size_t d, std::size_t count, std::size_t q, std::uint32_t seed)
        : dim(d), n(count), data(d * count), queries(d * q), nq(q) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> g(0.f, 1.f);
        for (auto& v : data) v = g(rng);
        for (auto& v : queries) v = g(rng);
    }
};

// Mean recall@k of the graph against exhaustive search over the same data.
double measure_recall(const Dataset& ds, vec::HnswIndex& idx,
                      const vec::FlatIndex& flat, std::size_t k, std::size_t ef) {
    double total = 0;
    for (std::size_t i = 0; i < ds.nq; ++i) {
        const float* q = ds.queries.data() + i * ds.dim;
        auto truth = flat.search(q, k);
        auto got = idx.search(q, k, ef);

        std::vector<std::uint32_t> truth_ids;
        for (const auto& t : truth) truth_ids.push_back(t.id);
        total += vec::recall_at_k(got, truth_ids.data(), k);
    }
    return total / static_cast<double>(ds.nq);
}

}  // namespace

int main() {
    // Trivial shapes first: an empty index, and one holding a single point.
    {
        vec::HnswIndex idx(4, 8);
        const float q[4] = {0, 0, 0, 0};
        CHECK(idx.search(q, 5, 10).empty());
        float v[4] = {1, 2, 3, 4};
        idx.add(v);
        auto r = idx.search(v, 5, 10);
        CHECK(r.size() == 1);
        CHECK(r[0].id == 0);
        CHECK_NEAR(r[0].dist, 0.0, 1e-6);
    }

    // Every inserted point must be its own nearest neighbour. If the graph is
    // disconnected anywhere, this is what catches it.
    {
        Dataset ds(32, 2000, 0, 7);
        vec::HnswIndex idx(ds.dim, ds.n);
        idx.add_many(ds.data.data(), ds.n);
        CHECK(idx.size() == ds.n);

        std::size_t found = 0;
        for (std::size_t i = 0; i < ds.n; i += 7) {
            auto r = idx.search(ds.data.data() + i * ds.dim, 1, 64);
            if (!r.empty() && r[0].id == i) ++found;
        }
        const std::size_t probed = (ds.n + 6) / 7;
        std::printf("  self-retrieval: %zu/%zu\n", found, probed);
        CHECK(found == probed);
    }

    // Results must be ordered nearest-first and free of duplicates -- a broken
    // visited set shows up here as the same id returned twice.
    {
        Dataset ds(16, 1500, 20, 21);
        vec::HnswIndex idx(ds.dim, ds.n);
        idx.add_many(ds.data.data(), ds.n);

        for (std::size_t i = 0; i < ds.nq; ++i) {
            auto r = idx.search(ds.queries.data() + i * ds.dim, 10, 64);
            CHECK(r.size() == 10);
            for (std::size_t j = 1; j < r.size(); ++j) {
                CHECK(r[j - 1].dist <= r[j].dist);
                for (std::size_t m = 0; m < j; ++m) CHECK(r[m].id != r[j].id);
            }
        }
    }

    // The bar from the plan: recall@10 >= 0.95 at ef=100.
    {
        Dataset ds(64, 10000, 200, 99);
        vec::HnswIndex idx(ds.dim, ds.n, vec::Metric::kL2, {16, 200, 42});
        idx.add_many(ds.data.data(), ds.n);

        vec::FlatIndex flat(ds.dim, ds.n);
        flat.add_many(ds.data.data(), ds.n);

        const double r10 = measure_recall(ds, idx, flat, 10, 100);
        std::printf("  recall@10 ef=100: %.4f   avg degree %.1f\n", r10,
                    idx.average_degree());
        CHECK(r10 >= 0.95);

        // Recall must rise monotonically with the beam width. If it does not,
        // the beam's termination condition is wrong.
        const double r_ef10 = measure_recall(ds, idx, flat, 10, 10);
        const double r_ef200 = measure_recall(ds, idx, flat, 10, 200);
        std::printf("  recall@10 ef=10: %.4f  ef=200: %.4f\n", r_ef10, r_ef200);
        CHECK(r_ef10 <= r10 + 1e-9);
        CHECK(r10 <= r_ef200 + 1e-9);
    }

    // A scratch-passing search must give byte-identical results to the
    // convenience overload; this is the contract the threaded benchmark rests on.
    {
        Dataset ds(24, 1000, 10, 5);
        vec::HnswIndex idx(ds.dim, ds.n);
        idx.add_many(ds.data.data(), ds.n);

        vec::Scratch s;
        for (std::size_t i = 0; i < ds.nq; ++i) {
            const float* q = ds.queries.data() + i * ds.dim;
            auto a = idx.search(q, 10, 50);
            auto b = idx.search(q, 10, 50, s);
            CHECK(a.size() == b.size());
            for (std::size_t j = 0; j < a.size(); ++j) CHECK(a[j].id == b[j].id);
        }
    }

    PASS();
}
