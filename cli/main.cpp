// Command line front end.
//
//   hnsw build  <vectors.fvecs> <out.idx> [M] [ef_construction]
//   hnsw query  <index.idx> <queries.fvecs> [k] [ef]
//   hnsw eval   <index.idx> <queries.fvecs> <truth.ivecs> [k]
//   hnsw info   <index.idx>
#include "vec/format.hpp"
#include "vec/fvecs.hpp"
#include "vec/hnsw.hpp"

#include "core/histogram.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using Clock = std::chrono::steady_clock;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  hnsw build <vectors.fvecs> <out.idx> [M=16] [ef_construction=200]\n"
                 "  hnsw query <index.idx> <queries.fvecs> [k=10] [ef=100]\n"
                 "  hnsw eval  <index.idx> <queries.fvecs> <truth.ivecs> [k=10]\n"
                 "  hnsw info  <index.idx>\n");
    return 2;
}

std::size_t arg_or(int argc, char** argv, int i, std::size_t fallback) {
    return i < argc ? static_cast<std::size_t>(std::strtoull(argv[i], nullptr, 10))
                    : fallback;
}

int cmd_build(int argc, char** argv) {
    if (argc < 4) return usage();
    const std::size_t M = arg_or(argc, argv, 4, 16);
    const std::size_t efc = arg_or(argc, argv, 5, 200);

    auto data = vec::read_fvecs(argv[2]);
    std::printf("building over %zu x %zu, M=%zu ef_construction=%zu\n", data.count,
                data.dim, M, efc);

    vec::HnswIndex idx(data.dim, data.count, vec::Metric::kL2, {M, efc, 42});
    const auto t0 = Clock::now();
    // Progress matters here: a 1M build runs for minutes and a silent binary is
    // indistinguishable from a hung one.
    const std::size_t step = data.count / 20 ? data.count / 20 : 1;
    for (std::size_t i = 0; i < data.count; ++i) {
        idx.add(data.row(i));
        if ((i + 1) % step == 0 || i + 1 == data.count) {
            const double el = std::chrono::duration<double>(Clock::now() - t0).count();
            std::printf("\r  %5.1f%%  %.0f vec/s  ",
                        100.0 * (i + 1) / data.count, (i + 1) / el);
            std::fflush(stdout);
        }
    }
    const double build_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("\nbuilt in %.1f s, avg degree %.1f\n", build_s, idx.average_degree());

    vec::save_index(idx, argv[3]);
    vec::MappedIndex m(argv[3]);
    std::printf("wrote %s (%.1f MiB)\n", argv[3], m.bytes() / 1048576.0);
    return 0;
}

int cmd_query(int argc, char** argv) {
    if (argc < 4) return usage();
    const std::size_t k = arg_or(argc, argv, 4, 10);
    const std::size_t ef = arg_or(argc, argv, 5, 100);

    const auto t0 = Clock::now();
    vec::MappedIndex idx(argv[2]);
    const double open_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    auto queries = vec::read_fvecs(argv[3]);
    if (queries.dim != idx.dim()) {
        std::fprintf(stderr, "dimension mismatch: index %zu, queries %zu\n",
                     idx.dim(), queries.dim);
        return 1;
    }
    std::printf("opened %zu vectors in %.3f ms\n\n", idx.size(), open_ms);

    vec::Scratch s;
    core::Histogram h;
    for (std::size_t i = 0; i < queries.count; ++i) {
        const auto q0 = Clock::now();
        auto r = idx.search(queries.row(i), k, ef, s);
        h.record(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - q0)
                     .count());
        if (i < 5) {  // show the first few, then just time the rest
            std::printf("query %zu:", i);
            for (const auto& n : r) std::printf("  %u (%.3f)", n.id, n.dist);
            std::printf("\n");
        }
    }
    std::printf("\n%zu queries, ef=%zu: %s\n", queries.count, ef,
                h.summary("ns").c_str());
    std::printf("%.0f QPS\n", 1e9 / h.mean());
    return 0;
}

int cmd_eval(int argc, char** argv) {
    if (argc < 5) return usage();
    const std::size_t k = arg_or(argc, argv, 5, 10);

    vec::MappedIndex idx(argv[2]);
    auto queries = vec::read_fvecs(argv[3]);
    auto truth = vec::read_ivecs(argv[4]);
    if (truth.count != queries.count) {
        std::fprintf(stderr, "truth has %zu rows, queries %zu\n", truth.count,
                     queries.count);
        return 1;
    }

    std::printf("%zu vectors, %zu queries, k=%zu\n\n", idx.size(), queries.count, k);
    std::printf("%6s %11s %12s %9s %9s\n", "ef", "recall", "QPS", "p50 us", "p99 us");

    vec::Scratch s;
    for (std::size_t ef : {10u, 20u, 40u, 60u, 100u, 200u, 400u, 800u}) {
        core::Histogram h;
        double rec = 0;
        for (std::size_t i = 0; i < queries.count; ++i) {
            const auto q0 = Clock::now();
            auto r = idx.search(queries.row(i), k, ef, s);
            h.record(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - q0)
                    .count());
            std::vector<std::uint32_t> t;
            for (std::size_t j = 0; j < k && j < truth.dim; ++j)
                t.push_back(static_cast<std::uint32_t>(truth.row(i)[j]));
            rec += vec::recall_at_k(r, t.data(), k);
        }
        std::printf("%6zu %11.4f %12.0f %9llu %9llu\n", ef, rec / queries.count,
                    1e9 / h.mean(), (unsigned long long)(h.percentile(0.50) / 1000),
                    (unsigned long long)(h.percentile(0.99) / 1000));
    }
    return 0;
}

int cmd_info(int argc, char** argv) {
    if (argc < 3) return usage();
    vec::MappedIndex idx(argv[2]);
    const auto& h = idx.header();
    std::printf("file           %s\n", argv[2]);
    std::printf("size           %.2f MiB\n", idx.bytes() / 1048576.0);
    std::printf("vectors        %u x %u (stride %u)\n", h.count, h.dim, h.stride);
    std::printf("metric         %s\n", h.metric == 0 ? "L2" : "inner product");
    std::printf("M / max_m0     %u / %u\n", h.M, h.max_m0);
    std::printf("ef_construction %u\n", h.ef_construction);
    std::printf("entry / levels %u / %d\n", h.entry, h.max_level);
    std::printf("bytes/vector   %.1f\n",
                static_cast<double>(idx.bytes()) / (h.count ? h.count : 1));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    try {
        if (cmd == "build") return cmd_build(argc, argv);
        if (cmd == "query") return cmd_query(argc, argv);
        if (cmd == "eval") return cmd_eval(argc, argv);
        if (cmd == "info") return cmd_info(argc, argv);
        return usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
