// The NEON kernels are only trustworthy if they agree with the scalar
// reference across every shape the beam search will hand them -- especially
// dimensions that are not multiples of 16, where the tail loops run.
#include "check.hpp"
#include "vec/distance.hpp"

#include <random>
#include <vector>

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Dimensions chosen to exercise every path: below one 16-wide block, exact
    // multiples, multiples plus a 4-wide remainder, and prime tails.
    const std::size_t dims[] = {1, 3, 4, 7, 15, 16, 17, 31, 32, 33,
                                63, 64, 96, 100, 128, 129, 384, 767, 960};

    for (std::size_t d : dims) {
        std::vector<float> a(d), b(d);
        for (std::size_t i = 0; i < d; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }

        CHECK_NEAR(vec::l2_sqr(a.data(), b.data(), d),
                   vec::l2_sqr_scalar(a.data(), b.data(), d), 1e-5);
        CHECK_NEAR(vec::inner_product(a.data(), b.data(), d),
                   vec::inner_product_scalar(a.data(), b.data(), d), 1e-5);

        // A vector is at distance zero from itself, and L2 is symmetric.
        CHECK_NEAR(vec::l2_sqr(a.data(), a.data(), d), 0.0, 1e-6);
        CHECK_NEAR(vec::l2_sqr(a.data(), b.data(), d),
                   vec::l2_sqr(b.data(), a.data(), d), 1e-6);
    }

    // Hand-computed case, so the whole thing is not just self-consistent.
    const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    CHECK_NEAR(vec::l2_sqr(x, y, 4), 30.0, 1e-6);          // 1+4+9+16
    CHECK_NEAR(vec::inner_product(x, x, 4), 30.0, 1e-6);

    PASS();
}
