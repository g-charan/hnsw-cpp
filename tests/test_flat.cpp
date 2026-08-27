// The flat index is the oracle everything else is scored against, so it has to
// be right before any of the graph work can be trusted.
#include "check.hpp"
#include "vec/flat.hpp"

#include <random>
#include <set>
#include <vector>

int main() {
    // Hand-placed points on a line: nearest neighbours are known by inspection.
    {
        vec::FlatIndex idx(1, 10);
        for (float v : {0.f, 1.f, 2.f, 3.f, 4.f, 5.f}) idx.add(&v);

        const float q = 2.2f;
        auto r = idx.search(&q, 3);
        CHECK(r.size() == 3);
        CHECK(r[0].id == 2);  // 2.0 is closest
        CHECK(r[1].id == 3);  // then 3.0
        CHECK(r[2].id == 1);  // then 1.0
        // Results must come back sorted nearest-first.
        CHECK(r[0].dist <= r[1].dist);
        CHECK(r[1].dist <= r[2].dist);
        CHECK_NEAR(r[0].dist, 0.04, 1e-5);  // (2.2-2.0)^2
    }

    // Every point is its own nearest neighbour, at distance zero.
    {
        const std::size_t dim = 37, n = 200;  // dim deliberately not a multiple of 4
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        std::vector<float> data(dim * n);
        for (auto& v : data) v = dist(rng);

        vec::FlatIndex idx(dim, n);
        idx.add_many(data.data(), n);
        CHECK(idx.size() == n);

        for (std::size_t i = 0; i < n; i += 17) {
            auto r = idx.search(data.data() + i * dim, 1);
            CHECK(r[0].id == i);
            CHECK_NEAR(r[0].dist, 0.0, 1e-5);
        }
    }

    // k larger than the corpus must return everything, not overrun.
    {
        vec::FlatIndex idx(2, 3);
        for (int i = 0; i < 3; ++i) { float v[2] = {float(i), float(i)}; idx.add(v); }
        const float q[2] = {0.f, 0.f};
        auto r = idx.search(q, 10);
        CHECK(r.size() == 3);
        CHECK(r[0].id == 0);
    }

    // recall_at_k: the definition every README number depends on.
    {
        std::vector<vec::Neighbor> got{{0.1f, 5}, {0.2f, 7}, {0.3f, 9}};
        const std::uint32_t perfect[3] = {5, 7, 9};
        const std::uint32_t half[3] = {5, 7, 42};
        const std::uint32_t none[3] = {1, 2, 3};
        CHECK_NEAR(vec::recall_at_k(got, perfect, 3), 1.0, 1e-9);
        CHECK_NEAR(vec::recall_at_k(got, half, 3), 2.0 / 3.0, 1e-9);
        CHECK_NEAR(vec::recall_at_k(got, none, 3), 0.0, 1e-9);
        // Order within the truth set must not matter.
        const std::uint32_t shuffled[3] = {9, 5, 7};
        CHECK_NEAR(vec::recall_at_k(got, shuffled, 3), 1.0, 1e-9);
    }

    PASS();
}
