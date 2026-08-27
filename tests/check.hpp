#pragma once
// Minimal assert harness. No framework: a test binary that returns non-zero on
// failure is all ctest needs, and CHECK carries the file/line and the failing
// expression, which is the only thing a framework would add here.
#include <cstdio>
#include <cstdlib>
#include <cmath>

inline int g_checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,       \
                         #cond);                                               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Relative comparison. The SIMD kernels sum in a different order than the
// scalar reference, so bit equality is the wrong bar; agreement to within
// float32's useful precision is the right one.
#define CHECK_NEAR(a, b, rel)                                                  \
    do {                                                                       \
        ++g_checks;                                                            \
        const double va = (a), vb = (b);                                       \
        const double scale = std::fmax(1.0, std::fmax(std::fabs(va),           \
                                                      std::fabs(vb)));         \
        if (std::fabs(va - vb) > (rel) * scale) {                              \
            std::fprintf(stderr,                                               \
                         "FAIL %s:%d: %s (%.9g) !~ %s (%.9g), rel=%g\n",       \
                         __FILE__, __LINE__, #a, va, #b, vb, (double)(rel));   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define PASS()                                                                 \
    do {                                                                       \
        std::printf("ok  %s (%d checks)\n", __FILE__, g_checks);               \
        return 0;                                                              \
    } while (0)
