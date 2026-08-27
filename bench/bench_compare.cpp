// Head to head against hnswlib, the reference implementation.
//
// Comparing raw QPS between two ANN indexes is meaningless unless recall is
// held equal -- any index can be made faster by returning worse answers. So
// both are swept across ef, and the comparison that matters is QPS at matched
// recall, read off the same row of the two tables.
//
//   bench_compare <dir> <prefix>     e.g. bench_compare data/sift sift
#include "core/histogram.hpp"
#include "vec/fvecs.hpp"
#include "vec/hnsw.hpp"

#include "hnswlib/hnswlib.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

struct Row {
    std::size_t ef;
    double recall, qps;
    std::uint64_t p50, p99;
};

void print_table(const char* label, const std::vector<Row>& rows, double build_s,
                 double mib) {
    std::printf("\n%s  (build %.1f s, %.0f MiB)\n", label, build_s, mib);
    std::printf("%6s %11s %12s %9s %9s\n", "ef", "recall@10", "QPS", "p50 us", "p99 us");
    for (const Row& r : rows)
        std::printf("%6zu %11.4f %12.0f %9llu %9llu\n", r.ef, r.recall, r.qps,
                    (unsigned long long)r.p50, (unsigned long long)r.p99);
}

// QPS this index reaches at or above a target recall -- the only fair
// single-number comparison between two approximate indexes.
double qps_at_recall(const std::vector<Row>& rows, double target) {
    double best = 0;
    for (const Row& r : rows)
        if (r.recall >= target && r.qps > best) best = r.qps;
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bench_compare <dir> [prefix]\n"); return 2; }
    const std::string dir = argv[1];
    const std::string pfx = argc > 2 ? argv[2] : "sift";

    auto base = vec::read_fvecs(dir + "/" + pfx + "_base.fvecs");
    auto query = vec::read_fvecs(dir + "/" + pfx + "_query.fvecs");
    auto gt = vec::read_ivecs(dir + "/" + pfx + "_groundtruth.ivecs");

    const std::size_t dim = base.dim, n = base.count, nq = query.count, k = 10;
    const std::size_t M = 16, efc = 200;
    std::printf("%s: %zu x %zu, %zu queries, k=%zu, M=%zu, ef_construction=%zu\n",
                pfx.c_str(), n, dim, nq, k, M, efc);

    auto truth_row = [&](std::size_t i) {
        std::vector<std::uint32_t> t;
        for (std::size_t j = 0; j < k; ++j)
            t.push_back(static_cast<std::uint32_t>(gt.row(i)[j]));
        return t;
    };

    const std::vector<std::size_t> efs = {10, 20, 40, 60, 100, 200, 400};

    // ---- this project --------------------------------------------------
    std::vector<Row> mine;
    double mine_build = 0, mine_mib = 0;
    {
        vec::HnswIndex idx(dim, n, vec::Metric::kL2, {M, efc, 42});
        const auto t0 = Clock::now();
        idx.add_many(base.data.data(), n);
        mine_build = std::chrono::duration<double>(Clock::now() - t0).count();
        mine_mib = idx.graph_bytes() / 1048576.0;

        vec::Scratch s;
        for (std::size_t ef : efs) {
            core::Histogram h;
            double rec = 0;
            for (std::size_t i = 0; i < nq; ++i) {
                const auto q0 = Clock::now();
                auto r = idx.search(query.row(i), k, ef, s);
                h.record(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             Clock::now() - q0).count());
                auto t = truth_row(i);
                rec += vec::recall_at_k(r, t.data(), k);
            }
            mine.push_back({ef, rec / nq, 1e9 / h.mean(), h.percentile(0.50) / 1000,
                            h.percentile(0.99) / 1000});
        }
    }

    // ---- hnswlib -------------------------------------------------------
    std::vector<Row> theirs;
    double their_build = 0, their_mib = 0;
    {
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> alg(&space, n, M, efc, 42);
        const auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i) alg.addPoint(base.row(i), i);
        their_build = std::chrono::duration<double>(Clock::now() - t0).count();
        their_mib = static_cast<double>(alg.size_links_level0_ * n +
                                        alg.size_data_per_element_ * n) / 1048576.0;

        for (std::size_t ef : efs) {
            alg.setEf(ef);
            core::Histogram h;
            double rec = 0;
            for (std::size_t i = 0; i < nq; ++i) {
                const auto q0 = Clock::now();
                auto res = alg.searchKnn(query.row(i), k);
                h.record(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             Clock::now() - q0).count());

                std::vector<vec::Neighbor> got;
                while (!res.empty()) {  // max-heap: furthest pops first
                    got.push_back({res.top().first,
                                   static_cast<std::uint32_t>(res.top().second)});
                    res.pop();
                }
                std::reverse(got.begin(), got.end());
                auto t = truth_row(i);
                rec += vec::recall_at_k(got, t.data(), k);
            }
            theirs.push_back({ef, rec / nq, 1e9 / h.mean(), h.percentile(0.50) / 1000,
                              h.percentile(0.99) / 1000});
        }
    }

    print_table("this project", mine, mine_build, mine_mib);
    print_table("hnswlib 0.8.0", theirs, their_build, their_mib);

    std::printf("\nQPS at matched recall\n%12s %14s %14s %10s\n", "recall>=",
                "this project", "hnswlib", "ratio");
    for (double target : {0.90, 0.95, 0.99}) {
        const double a = qps_at_recall(mine, target), b = qps_at_recall(theirs, target);
        if (a == 0 || b == 0) {
            std::printf("%12.2f %14s %14s %10s\n", target,
                        a ? "-" : "not reached", b ? "-" : "not reached", "-");
        } else {
            std::printf("%12.2f %14.0f %14.0f %9.2fx\n", target, a, b, a / b);
        }
    }
    std::printf("\nbuild: %.1f s vs %.1f s  (%.2fx)\n", mine_build, their_build,
                their_build / mine_build);
    return 0;
}
