#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace core {

// Latency histogram with logarithmic bucketing, in the shape HdrHistogram uses.
//
// Recording must not perturb what it measures, so a record is an index
// computation and an increment -- no allocation, no branch on a growing vector.
// Values are bucketed by exponent with kSubBuckets linear slots inside each
// exponent, which holds relative error under 1/kSubBuckets across the whole
// range instead of the absolute error a fixed-width bucketing would give.
//
// Tail percentiles are the point. A mean latency hides exactly the stalls that
// matter, so p99 and p99.9 are what get reported.
class Histogram {
public:
    static constexpr int kSubBucketBits = 7;
    static constexpr int kSubBuckets = 1 << kSubBucketBits;  // 128
    static constexpr int kBuckets = 64 - kSubBucketBits + 1;

    Histogram() : counts_(static_cast<std::size_t>(kBuckets) * kSubBuckets, 0) {}

    void record(std::uint64_t value) {
        counts_[bucket_index(value)]++;
        count_++;
        sum_ += value;
        if (value > max_) max_ = value;
        if (value < min_) min_ = value;
    }

    void merge(const Histogram& o) {
        for (std::size_t i = 0; i < counts_.size(); ++i) counts_[i] += o.counts_[i];
        count_ += o.count_;
        sum_ += o.sum_;
        max_ = std::max(max_, o.max_);
        min_ = std::min(min_, o.min_);
    }

    void reset() {
        std::fill(counts_.begin(), counts_.end(), 0);
        count_ = 0;
        sum_ = 0;
        max_ = 0;
        min_ = UINT64_MAX;
    }

    std::uint64_t count() const { return count_; }
    std::uint64_t max() const { return max_; }
    std::uint64_t min() const { return count_ ? min_ : 0; }
    double mean() const { return count_ ? static_cast<double>(sum_) / count_ : 0.0; }

    // Smallest recorded value at or below which `p` of samples fall (p in [0,1]).
    std::uint64_t percentile(double p) const {
        if (count_ == 0) return 0;
        const auto target = static_cast<std::uint64_t>(
            std::ceil(p * static_cast<double>(count_)));
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < counts_.size(); ++i) {
            seen += counts_[i];
            if (seen >= target) return value_at(i);
        }
        return max_;
    }

    std::string summary(const char* unit = "ns") const {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "n=%llu  min=%llu  p50=%llu  p90=%llu  p99=%llu  "
                      "p99.9=%llu  max=%llu %s",
                      (unsigned long long)count_, (unsigned long long)min(),
                      (unsigned long long)percentile(0.50),
                      (unsigned long long)percentile(0.90),
                      (unsigned long long)percentile(0.99),
                      (unsigned long long)percentile(0.999),
                      (unsigned long long)max_, unit);
        return std::string(buf);
    }

private:
    // Index = (exponent block) * kSubBuckets + (linear slot within the block).
    static std::size_t bucket_index(std::uint64_t v) {
        if (v < kSubBuckets) return static_cast<std::size_t>(v);
        const int msb = 63 - __builtin_clzll(v);
        const int shift = msb - kSubBucketBits;
        const auto sub = static_cast<std::size_t>((v >> shift) & (kSubBuckets - 1));
        return static_cast<std::size_t>(shift + 1) * kSubBuckets + sub;
    }

    // Lower edge of a bucket -- the inverse of bucket_index.
    static std::uint64_t value_at(std::size_t idx) {
        const auto block = static_cast<int>(idx / kSubBuckets);
        const auto sub = static_cast<std::uint64_t>(idx % kSubBuckets);
        if (block == 0) return sub;
        const int shift = block - 1;
        return (sub | static_cast<std::uint64_t>(kSubBuckets)) << shift;
    }

    std::vector<std::uint64_t> counts_;
    std::uint64_t count_ = 0;
    std::uint64_t sum_ = 0;
    std::uint64_t max_ = 0;
    std::uint64_t min_ = UINT64_MAX;
};

}  // namespace core
