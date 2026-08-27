// Does the NEON path actually earn its complexity? This is the measurement the
// README quotes, at the two dimensions that matter: 128 (SIFT1M, the recall
// benchmark) and 384 (MiniLM, the WASM demo corpus).
#include "core/bench.hpp"
#include "vec/distance.hpp"

#include <cstdio>
#include <random>
#include <vector>

namespace {

// Distances are called against many different vectors, not one hot pair, so the
// benchmark walks a buffer larger than L2 to keep the memory behaviour honest.
struct Corpus {
    std::size_t dim;
    std::size_t n;
    std::vector<float> data;
    std::vector<float> query;

    Corpus(std::size_t d, std::size_t count) : dim(d), n(count), data(d * count), query(d) {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : data) v = dist(rng);
        for (auto& v : query) v = dist(rng);
    }
};

void compare(std::size_t dim) {
    Corpus c(dim, 20000);  // 20k x dim x 4B -> well past L2 at both dims
    const std::size_t ops = c.n;

    auto scalar = core::run("l2 scalar d=" + std::to_string(dim), 50, ops, [&] {
        float acc = 0;
        for (std::size_t i = 0; i < c.n; ++i)
            acc += vec::l2_sqr_scalar(c.query.data(), c.data.data() + i * dim, dim);
        core::keep(acc);
    });

    auto simd = core::run("l2 NEON   d=" + std::to_string(dim), 50, ops, [&] {
        float acc = 0;
        for (std::size_t i = 0; i < c.n; ++i)
            acc += vec::l2_sqr(c.query.data(), c.data.data() + i * dim, dim);
        core::keep(acc);
    });

    scalar.print();
    simd.print();
    std::printf("%-28s %10.2fx\n\n", "  speedup",
                scalar.ns_per_op / simd.ns_per_op);
}

}  // namespace

int main() {
    std::printf("distance kernels, %s\n\n", VEC_NEON ? "NEON available" : "scalar only");
    compare(128);
    compare(384);
    return 0;
}
