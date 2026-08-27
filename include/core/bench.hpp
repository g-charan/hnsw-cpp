#pragma once

#include "core/histogram.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>

namespace core {

// Keeps a computed value from being optimised away. Without it the compiler is
// entirely within its rights to delete a benchmark loop whose result is unused,
// and the kernel "runs" in zero nanoseconds.
template <class T>
inline void keep(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

inline void clobber() { asm volatile("" : : : "memory"); }

struct Result {
    std::string name;
    Histogram batches;    // ns per batch
    double ns_per_op = 0;
    double ops_per_sec = 0;
    std::uint64_t total_ops = 0;

    void print() const {
        std::printf("%-28s %10.2f ns/op  %12.2f Mops/s   [%s]\n", name.c_str(),
                    ns_per_op, ops_per_sec / 1e6, batches.summary("ns").c_str());
    }
};

// Times `batches` runs of a callable that performs `ops_per_batch` operations.
//
// Kernels here take tens of nanoseconds, which is the same order as the clock
// read itself, so timing individual calls would measure the timer. Batching
// amortises that away; the histogram then describes batch-to-batch variance,
// which is what exposes a noisy machine.
template <class F>
Result run(const std::string& name, std::size_t batches, std::size_t ops_per_batch,
           F&& f) {
    using Clock = std::chrono::steady_clock;

    for (std::size_t i = 0; i < 3; ++i) f();  // warm caches and branch predictors

    Result r;
    r.name = name;
    std::uint64_t total_ns = 0;

    for (std::size_t i = 0; i < batches; ++i) {
        const auto t0 = Clock::now();
        f();
        const auto t1 = Clock::now();
        const auto ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        r.batches.record(ns);
        total_ns += ns;
    }

    r.total_ops = static_cast<std::uint64_t>(batches) * ops_per_batch;
    r.ns_per_op = static_cast<double>(total_ns) / static_cast<double>(r.total_ops);
    r.ops_per_sec = r.ns_per_op > 0 ? 1e9 / r.ns_per_op : 0;
    return r;
}

}  // namespace core
