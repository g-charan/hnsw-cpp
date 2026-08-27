// Recall measured against a dataset whose ground truth someone else computed.
//
// Every other test scores the graph against this project's own flat index,
// which would keep passing if both shared a misunderstanding of the data. SIFT
// ships ground truth from an independent exhaustive search, so this is the one
// check that can catch that class of error -- and it does double duty as a real
// file for the .fvecs reader, which synthetic round-trips cannot.
//
// Skips (exit 77) when data/ has not been fetched; see tools/get_dataset.sh.
#include "check.hpp"
#include "vec/flat.hpp"
#include "vec/fvecs.hpp"
#include "vec/hnsw.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "data/siftsmall";
    const std::string prefix = argc > 2 ? argv[2] : "siftsmall";

    vec::VecFile<float> base, query;
    vec::VecFile<std::int32_t> gt;
    try {
        base = vec::read_fvecs(dir + "/" + prefix + "_base.fvecs");
        query = vec::read_fvecs(dir + "/" + prefix + "_query.fvecs");
        gt = vec::read_ivecs(dir + "/" + prefix + "_groundtruth.ivecs");
    } catch (const std::exception& e) {
        std::printf("skip: %s (run tools/get_dataset.sh siftsmall)\n", e.what());
        return 77;
    }

    CHECK(base.dim == query.dim);
    CHECK(gt.count == query.count);
    CHECK(base.count > 0);

    const std::size_t k = 10;
    auto truth_row = [&](std::size_t i) {
        std::vector<std::uint32_t> t;
        for (std::size_t j = 0; j < k; ++j)
            t.push_back(static_cast<std::uint32_t>(gt.row(i)[j]));
        return t;
    };

    // Exhaustive search must reproduce the shipped ground truth exactly. If this
    // slips, the reader is misparsing and every recall figure is fiction.
    {
        vec::FlatIndex flat(base.dim, base.count);
        flat.add_many(base.data.data(), base.count);
        double agree = 0;
        for (std::size_t i = 0; i < query.count; ++i) {
            auto r = flat.search(query.row(i), k);
            auto t = truth_row(i);
            agree += vec::recall_at_k(r, t.data(), k);
        }
        agree /= static_cast<double>(query.count);
        std::printf("  flat vs shipped ground truth: %.4f\n", agree);
        CHECK_NEAR(agree, 1.0, 1e-9);
    }

    // The graph, against that same independent truth.
    {
        vec::HnswIndex idx(base.dim, base.count, vec::Metric::kL2, {16, 200, 42});
        idx.add_many(base.data.data(), base.count);

        double prev = -1.0;
        for (std::size_t ef : {10u, 50u, 100u, 200u}) {
            double rec = 0;
            for (std::size_t i = 0; i < query.count; ++i) {
                auto r = idx.search(query.row(i), k, ef);
                auto t = truth_row(i);
                rec += vec::recall_at_k(r, t.data(), k);
            }
            rec /= static_cast<double>(query.count);
            std::printf("  ef=%-4zu recall@10 %.4f\n", ef, rec);
            CHECK(rec >= prev - 1e-9);  // must not fall as the beam widens
            prev = rec;
            if (ef == 100) CHECK(rec >= 0.95);  // the bar, on real data
        }
        std::printf("  avg degree %.1f\n", idx.average_degree());
    }

    PASS();
}
