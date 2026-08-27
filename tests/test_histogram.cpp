#include "check.hpp"
#include "core/histogram.hpp"

#include <cstdint>
#include <vector>

int main() {
    // Percentiles over a known uniform distribution. Bucketing is lossy by
    // design, so the bar is relative error under 1/kSubBuckets, not exactness.
    {
        core::Histogram h;
        for (std::uint64_t v = 1; v <= 10000; ++v) h.record(v);
        CHECK(h.count() == 10000);
        CHECK(h.max() == 10000);
        CHECK(h.min() == 1);
        CHECK_NEAR(h.percentile(0.50), 5000.0, 0.02);
        CHECK_NEAR(h.percentile(0.90), 9000.0, 0.02);
        CHECK_NEAR(h.percentile(0.99), 9900.0, 0.02);
        CHECK_NEAR(h.mean(), 5000.5, 0.01);
    }

    // Small values are stored exactly: the first block is linear, and
    // sub-microsecond latencies are the ones that must not blur.
    {
        core::Histogram h;
        for (std::uint64_t v = 0; v < 128; ++v) h.record(v);
        CHECK(h.percentile(1.0) == 127);
        CHECK(h.min() == 0);
    }

    // A tail spike must show up in p99.9 and max, and must not move p50 --
    // this is the whole reason the histogram exists.
    {
        core::Histogram h;
        for (int i = 0; i < 9990; ++i) h.record(100);
        for (int i = 0; i < 10; ++i) h.record(1000000);
        CHECK_NEAR(h.percentile(0.50), 100.0, 0.02);
        CHECK(h.percentile(0.9999) >= 900000);
        CHECK(h.max() == 1000000);
    }

    // merge must be equivalent to having recorded into one histogram.
    {
        core::Histogram a, b, both;
        for (std::uint64_t v = 1; v <= 500; ++v) { a.record(v); both.record(v); }
        for (std::uint64_t v = 501; v <= 1000; ++v) { b.record(v); both.record(v); }
        a.merge(b);
        CHECK(a.count() == both.count());
        CHECK(a.max() == both.max());
        CHECK(a.percentile(0.50) == both.percentile(0.50));
        CHECK(a.percentile(0.99) == both.percentile(0.99));
    }

    { core::Histogram h; CHECK(h.percentile(0.5) == 0); CHECK(h.count() == 0); }

    PASS();
}
