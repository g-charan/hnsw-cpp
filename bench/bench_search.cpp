// The headline measurement: recall against exhaustive search, and the query
// throughput bought at each recall level.
//
//   bench_search                 synthetic 100k x 128 gaussian
//   bench_search <sift_dir>      real SIFT1M, using its shipped ground truth
//
// Queries are timed individually. A query costs tens of microseconds, far above
// the cost of reading the clock, so per-query timing is honest here and gives
// real tail percentiles rather than a batch average.
#include "core/histogram.hpp"
#include "vec/flat.hpp"
#include "vec/fvecs.hpp"
#include "vec/hnsw.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Data {
    std::size_t dim = 0, n = 0, nq = 0;
    std::vector<float> base, query;
    std::vector<std::uint32_t> truth;  // nq * k_truth
    std::size_t k_truth = 0;
    std::string name;
};

Data synthetic(std::size_t dim, std::size_t n, std::size_t nq) {
    Data d;
    d.dim = dim; d.n = n; d.nq = nq;
    d.name = "synthetic gaussian " + std::to_string(n) + "x" + std::to_string(dim);
    d.base.resize(dim * n);
    d.query.resize(dim * nq);
    std::mt19937 rng(1234);
    std::normal_distribution<float> g(0.f, 1.f);
    for (auto& v : d.base) v = g(rng);
    for (auto& v : d.query) v = g(rng);
    return d;
}

Data load_sift(const std::string& dir) {
    Data d;
    auto base = vec::read_fvecs(dir + "/sift_base.fvecs");
    auto query = vec::read_fvecs(dir + "/sift_query.fvecs");
    auto gt = vec::read_ivecs(dir + "/sift_groundtruth.ivecs");

    d.dim = base.dim; d.n = base.count; d.nq = query.count;
    d.base = std::move(base.data);
    d.query = std::move(query.data);
    d.k_truth = gt.dim;
    d.truth.assign(gt.data.begin(), gt.data.end());
    d.name = "SIFT1M " + std::to_string(d.n) + "x" + std::to_string(d.dim);
    return d;
}

// Ground truth by exhaustive search, for the synthetic case.
void compute_truth(Data& d, std::size_t k) {
    vec::FlatIndex flat(d.dim, d.n);
    flat.add_many(d.base.data(), d.n);
    d.k_truth = k;
    d.truth.resize(d.nq * k);
    for (std::size_t i = 0; i < d.nq; ++i) {
        auto r = flat.search(d.query.data() + i * d.dim, k);
        for (std::size_t j = 0; j < k; ++j) {
            d.truth[i * k + j] = j < r.size() ? r[j].id : 0xFFFFFFFFu;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t k = 10;
    Data d = argc > 1 ? load_sift(argv[1]) : synthetic(128, 100000, 1000);

    std::printf("dataset: %s, %zu queries, k=%zu\n\n", d.name.c_str(), d.nq, k);

    if (d.truth.empty()) {
        std::printf("computing ground truth by exhaustive search...\n");
        compute_truth(d, k);
    }

    // ---- baseline: exhaustive search -------------------------------------
    vec::FlatIndex flat(d.dim, d.n);
    flat.add_many(d.base.data(), d.n);

    core::Histogram flat_h;
    for (std::size_t i = 0; i < d.nq; ++i) {
        const auto t0 = Clock::now();
        auto r = flat.search(d.query.data() + i * d.dim, k);
        const auto t1 = Clock::now();
        flat_h.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        if (r.empty()) std::abort();
    }
    const double flat_qps = 1e9 / flat_h.mean();
    std::printf("brute force   %8.0f QPS   p50 %7llu us  p99 %7llu us\n\n", flat_qps,
                (unsigned long long)(flat_h.percentile(0.50) / 1000),
                (unsigned long long)(flat_h.percentile(0.99) / 1000));

    // ---- graph index ------------------------------------------------------
    vec::HnswIndex idx(d.dim, d.n, vec::Metric::kL2, {16, 200, 42});
    const auto b0 = Clock::now();
    idx.add_many(d.base.data(), d.n);
    const auto b1 = Clock::now();
    const double build_s = std::chrono::duration<double>(b1 - b0).count();

    std::printf("build: %.1f s  (%.0f vectors/s)  graph %.1f MiB  avg degree %.1f\n\n",
                build_s, d.n / build_s, idx.graph_bytes() / 1048576.0,
                idx.average_degree());

    std::printf("%6s %10s %12s %10s %10s %10s\n", "ef", "recall@10", "QPS",
                "p50 us", "p99 us", "speedup");
    for (std::size_t ef : {10, 20, 40, 60, 100, 200, 400}) {
        core::Histogram h;
        double recall = 0;
        vec::HnswIndex::Scratch s;
        for (std::size_t i = 0; i < d.nq; ++i) {
            const float* q = d.query.data() + i * d.dim;
            const auto t0 = Clock::now();
            auto r = idx.search(q, k, ef, s);
            const auto t1 = Clock::now();
            h.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            recall += vec::recall_at_k(r, d.truth.data() + i * d.k_truth, k);
        }
        recall /= static_cast<double>(d.nq);
        const double qps = 1e9 / h.mean();
        std::printf("%6zu %10.4f %12.0f %10llu %10llu %9.1fx\n", ef, recall, qps,
                    (unsigned long long)(h.percentile(0.50) / 1000),
                    (unsigned long long)(h.percentile(0.99) / 1000),
                    qps / flat_qps);
    }
    return 0;
}
