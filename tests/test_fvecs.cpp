// Reading the benchmark datasets wrong would silently corrupt every recall
// number, so the reader gets round-tripped and its failure modes pinned down.
#include "check.hpp"
#include "vec/fvecs.hpp"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main() {
    const std::string base = "/tmp/hnsw_test_fvecs_";
    const std::string fpath = base + "a.fvecs";
    const std::string ipath = base + "a.ivecs";

    // Round-trip floats.
    {
        const std::size_t dim = 13, n = 50;
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> dist(-5.f, 5.f);
        std::vector<float> data(dim * n);
        for (auto& v : data) v = dist(rng);

        vec::write_xvecs(fpath, data.data(), n, dim);
        auto got = vec::read_fvecs(fpath);
        CHECK(got.dim == dim);
        CHECK(got.count == n);
        for (std::size_t i = 0; i < dim * n; ++i) CHECK(got.data[i] == data[i]);
        CHECK(got.row(2)[0] == data[2 * dim]);
    }

    // Round-trip ints (ground-truth files are ivecs).
    {
        const std::size_t dim = 4, n = 7;
        std::vector<std::int32_t> data(dim * n);
        for (std::size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<std::int32_t>(i * 3);
        vec::write_xvecs(ipath, data.data(), n, dim);
        auto got = vec::read_ivecs(ipath);
        CHECK(got.dim == dim);
        CHECK(got.count == n);
        for (std::size_t i = 0; i < data.size(); ++i) CHECK(got.data[i] == data[i]);
    }

    // A truncated file must be rejected, not read as garbage. This is the
    // failure that would otherwise show up as an inexplicable recall drop.
    {
        const std::string tpath = base + "trunc.fvecs";
        std::vector<float> data(8 * 4, 1.0f);
        vec::write_xvecs(tpath, data.data(), 8, 4);
        // Chop three bytes off the end.
        std::FILE* f = std::fopen(tpath.c_str(), "rb");
        std::vector<char> buf(1024);
        const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        f = std::fopen(tpath.c_str(), "wb");
        std::fwrite(buf.data(), 1, got - 3, f);
        std::fclose(f);

        bool threw = false;
        try { vec::read_fvecs(tpath); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        std::remove(tpath.c_str());
    }

    // A missing file must throw rather than return an empty set that would
    // quietly benchmark nothing.
    {
        bool threw = false;
        try { vec::read_fvecs(base + "does_not_exist.fvecs"); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    std::remove(fpath.c_str());
    std::remove(ipath.c_str());
    PASS();
}
