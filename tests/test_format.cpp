// A saved index must answer queries identically to the one it was saved from,
// and a damaged file must be rejected at open rather than crashing mid-search.
#include "check.hpp"
#include "vec/format.hpp"
#include "vec/hnsw.hpp"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main() {
    const std::string path = "/tmp/hnsw_test_index.bin";
    const std::size_t dim = 48, n = 3000, nq = 100;

    std::mt19937 rng(4242);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> base(dim * n), queries(dim * nq);
    for (auto& v : base) v = g(rng);
    for (auto& v : queries) v = g(rng);

    vec::HnswIndex idx(dim, n, vec::Metric::kL2, {16, 200, 7});
    idx.add_many(base.data(), n);
    vec::save_index(idx, path);

    {
        vec::MappedIndex m(path);
        CHECK(m.size() == n);
        CHECK(m.dim() == dim);
        CHECK(m.header().magic == vec::kMagic);
        CHECK(m.view().entry == idx.entry_point());
        CHECK(m.view().max_level == idx.max_level());

        // Vector rows must be 16-byte aligned in the mapping, or the NEON loads
        // in the distance kernels straddle.
        CHECK(reinterpret_cast<std::uintptr_t>(m.view().vectors) % 16 == 0);

        // Identical results, not merely similar ones: same ids, same distances.
        vec::Scratch s1, s2;
        for (std::size_t i = 0; i < nq; ++i) {
            const float* q = queries.data() + i * dim;
            for (std::size_t ef : {10u, 64u, 200u}) {
                auto a = idx.search(q, 10, ef, s1);
                auto b = m.search(q, 10, ef, s2);
                CHECK(a.size() == b.size());
                for (std::size_t j = 0; j < a.size(); ++j) {
                    CHECK(a[j].id == b[j].id);
                    CHECK(a[j].dist == b[j].dist);
                }
            }
        }
    }

    // Truncation must be caught at open. This is the failure that would
    // otherwise read neighbour ids out of unmapped memory.
    {
        const std::string bad = "/tmp/hnsw_test_trunc.bin";
        std::FILE* in = std::fopen(path.c_str(), "rb");
        std::fseek(in, 0, SEEK_END);
        const long full = std::ftell(in);
        std::fseek(in, 0, SEEK_SET);
        std::vector<char> buf(static_cast<std::size_t>(full));
        CHECK(std::fread(buf.data(), 1, buf.size(), in) == buf.size());
        std::fclose(in);

        std::FILE* out = std::fopen(bad.c_str(), "wb");
        std::fwrite(buf.data(), 1, buf.size() / 2, out);  // half a file
        std::fclose(out);

        bool threw = false;
        try { vec::MappedIndex m(bad); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        std::remove(bad.c_str());
    }

    // A file that is not an index at all must be rejected on magic.
    {
        const std::string bad = "/tmp/hnsw_test_garbage.bin";
        std::FILE* out = std::fopen(bad.c_str(), "wb");
        std::vector<char> junk(4096, 0x5A);
        std::fwrite(junk.data(), 1, junk.size(), out);
        std::fclose(out);

        bool threw = false;
        try { vec::MappedIndex m(bad); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        std::remove(bad.c_str());
    }

    std::remove(path.c_str());
    PASS();
}
