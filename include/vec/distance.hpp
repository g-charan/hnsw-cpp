#pragma once

#include <cstddef>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VEC_NEON 1
#else
#define VEC_NEON 0
#endif

namespace vec {

// Distance kernels over float32 vectors.
//
// Every kernel has a scalar reference and, on aarch64, a NEON path. They are
// header-only and marked inline on purpose: the beam search calls these once
// per candidate edge, and letting the compiler inline the kernel into that loop
// is worth more than any instruction-level tuning inside it.
//
// The NEON paths carry four independent accumulators. A single accumulator
// serialises on the FMA latency chain (~4 cycles on Apple cores) and leaves the
// pipeline mostly idle; four in flight keep it fed. Because they sum in a
// different order than the scalar path, results differ in the low bits -- the
// tests compare with a relative epsilon, never for equality.

// ---------------------------------------------------------------- scalar ----

inline float l2_sqr_scalar(const float* a, const float* b, std::size_t d) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < d; ++i) {
        const float t = a[i] - b[i];
        acc += t * t;
    }
    return acc;
}

inline float inner_product_scalar(const float* a, const float* b, std::size_t d) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < d; ++i) acc += a[i] * b[i];
    return acc;
}

// ------------------------------------------------------------------ NEON ----

#if VEC_NEON

inline float l2_sqr_neon(const float* a, const float* b, std::size_t d) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i +  0), vld1q_f32(b + i +  0));
        const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
        s2 = vfmaq_f32(s2, d2, d2);
        s3 = vfmaq_f32(s3, d3, d3);
    }
    for (; i + 4 <= d; i += 4) {
        const float32x4_t dd = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        s0 = vfmaq_f32(s0, dd, dd);
    }

    float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < d; ++i) {
        const float t = a[i] - b[i];
        acc += t * t;
    }
    return acc;
}

inline float inner_product_neon(const float* a, const float* b, std::size_t d) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i +  0), vld1q_f32(b + i +  0));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= d; i += 4) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < d; ++i) acc += a[i] * b[i];
    return acc;
}

#endif  // VEC_NEON

// -------------------------------------------------------------- dispatch ----
//
// Compile-time dispatch. There is no runtime CPU check because there is nothing
// to choose between: NEON is baseline on every aarch64 target, and the scalar
// path exists for portability and as the test oracle.
//
// No AVX2 path: there is no x86 machine here to test one on, and an
// untested SIMD kernel is a correctness risk, not a feature. Add it behind
// __AVX2__ with the same four-accumulator shape when there is hardware to
// verify it against.

inline float l2_sqr(const float* a, const float* b, std::size_t d) {
#if VEC_NEON
    return l2_sqr_neon(a, b, d);
#else
    return l2_sqr_scalar(a, b, d);
#endif
}

inline float inner_product(const float* a, const float* b, std::size_t d) {
#if VEC_NEON
    return inner_product_neon(a, b, d);
#else
    return inner_product_scalar(a, b, d);
#endif
}

// Distance metrics, as the value HNSW orders candidates by: smaller is closer.
enum class Metric { kL2, kInnerProduct };

inline float distance(Metric m, const float* a, const float* b, std::size_t d) {
    return m == Metric::kL2 ? l2_sqr(a, b, d) : 1.0f - inner_product(a, b, d);
}

}  // namespace vec
