#pragma once

#include "core/mmap_file.hpp"
#include "vec/hnsw.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vec {

// On-disk index format.
//
// The whole design goal is that opening an index costs one mmap and no parsing:
// every block is stored in exactly the layout the search code already expects,
// so GraphView is built by pointing at offsets rather than by decoding anything.
// That is also why upper-layer lists carry an explicit per-node offset array
// instead of pointers -- a pointer would need a fix-up pass over every node on
// load, which is precisely the O(n) work this avoids.
//
// Blocks are padded to 64-byte boundaries. mmap hands back a page-aligned base,
// so an aligned offset means an aligned address, and the NEON loads over the
// vector block never straddle a cache line.
//
//   [ header 128B ][ vectors ][ level0 ][ upper_off ][ upper ][ levels ]

struct IndexHeader {
    std::uint32_t magic;    // kMagic
    std::uint32_t version;  // kVersion
    std::uint32_t dim;
    std::uint32_t stride;
    std::uint32_t count;
    std::uint32_t metric;
    std::uint32_t M;
    std::uint32_t max_m0;
    std::uint32_t ef_construction;
    std::uint32_t entry;
    std::int32_t max_level;
    std::uint32_t reserved;
    std::uint64_t vectors_off, vectors_len;
    std::uint64_t level0_off, level0_len;
    std::uint64_t upper_off_off, upper_off_len;
    std::uint64_t upper_off, upper_len;
    std::uint64_t levels_off, levels_len;
};
static_assert(sizeof(IndexHeader) == 128, "header must stay 128 bytes");

inline constexpr std::uint32_t kMagic = 0x57534E48u;  // "HNSW" little-endian
inline constexpr std::uint32_t kVersion = 1u;

namespace detail {

inline std::uint64_t align64(std::uint64_t v) { return (v + 63u) & ~std::uint64_t{63}; }

inline void pad_to(std::FILE* f, std::uint64_t target) {
    const long cur = std::ftell(f);
    if (cur < 0) throw std::runtime_error("ftell failed");
    static const char zeros[64] = {};
    std::uint64_t need = target - static_cast<std::uint64_t>(cur);
    while (need > 0) {
        const std::size_t chunk = need > 64 ? 64 : static_cast<std::size_t>(need);
        if (std::fwrite(zeros, 1, chunk, f) != chunk)
            throw std::runtime_error("short write while padding");
        need -= chunk;
    }
}

inline void write_block(std::FILE* f, const void* data, std::uint64_t bytes) {
    if (bytes == 0) return;
    if (std::fwrite(data, 1, bytes, f) != bytes)
        throw std::runtime_error("short write");
}

}  // namespace detail

// Writing happens once and is not on any hot path, so it uses ordinary buffered
// writes rather than a second mapping.
inline void save_index(const HnswIndex& idx, const std::string& path) {
    const GraphView g = idx.view();

    IndexHeader h{};
    h.magic = kMagic;
    h.version = kVersion;
    h.dim = static_cast<std::uint32_t>(g.dim);
    h.stride = static_cast<std::uint32_t>(g.stride);
    h.count = static_cast<std::uint32_t>(g.count);
    h.metric = static_cast<std::uint32_t>(g.metric == Metric::kL2 ? 0 : 1);
    h.M = static_cast<std::uint32_t>(g.M);
    h.max_m0 = static_cast<std::uint32_t>(g.max_m0);
    h.ef_construction = static_cast<std::uint32_t>(idx.params().ef_construction);
    h.entry = g.entry;
    h.max_level = g.max_level;

    const std::uint64_t n = g.count;
    h.vectors_len = n * g.stride * sizeof(float);
    h.level0_len = n * (g.max_m0 + 1) * sizeof(std::uint32_t);
    h.upper_off_len = n * sizeof(std::uint64_t);
    h.upper_len = idx.upper_data().size() * sizeof(std::uint32_t);
    h.levels_len = n * sizeof(std::int32_t);

    h.vectors_off = detail::align64(sizeof(IndexHeader));
    h.level0_off = detail::align64(h.vectors_off + h.vectors_len);
    h.upper_off_off = detail::align64(h.level0_off + h.level0_len);
    h.upper_off = detail::align64(h.upper_off_off + h.upper_off_len);
    h.levels_off = detail::align64(h.upper_off + h.upper_len);

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open for write: " + path);

    try {
        detail::write_block(f, &h, sizeof(h));
        detail::pad_to(f, h.vectors_off);
        detail::write_block(f, g.vectors, h.vectors_len);
        detail::pad_to(f, h.level0_off);
        detail::write_block(f, g.level0, h.level0_len);
        detail::pad_to(f, h.upper_off_off);
        detail::write_block(f, idx.upper_offsets().data(), h.upper_off_len);
        detail::pad_to(f, h.upper_off);
        detail::write_block(f, idx.upper_data().data(), h.upper_len);
        detail::pad_to(f, h.levels_off);
        detail::write_block(f, idx.levels().data(), h.levels_len);
    } catch (...) {
        std::fclose(f);
        throw;
    }
    if (std::fclose(f) != 0) throw std::runtime_error("close failed: " + path);
}

// An index read straight off disk. Owns the mapping, nothing else.
class MappedIndex {
public:
    explicit MappedIndex(const std::string& path) : file_(path) {
        if (file_.size() < sizeof(IndexHeader))
            throw std::runtime_error("index too small to hold a header: " + path);

        IndexHeader h{};
        std::memcpy(&h, file_.data(), sizeof(h));

        if (h.magic != kMagic)
            throw std::runtime_error("not an hnsw index (bad magic): " + path);
        if (h.version != kVersion)
            throw std::runtime_error("index version " + std::to_string(h.version) +
                                     ", expected " + std::to_string(kVersion));

        // Every block must lie inside the mapping. A truncated file has to fail
        // here, loudly, rather than somewhere inside the beam search.
        auto check = [&](std::uint64_t off, std::uint64_t len, const char* what) {
            if (off + len > file_.size())
                throw std::runtime_error(std::string("index block '") + what +
                                         "' runs past end of file: " + path);
        };
        check(h.vectors_off, h.vectors_len, "vectors");
        check(h.level0_off, h.level0_len, "level0");
        check(h.upper_off_off, h.upper_off_len, "upper_off");
        check(h.upper_off, h.upper_len, "upper");
        check(h.levels_off, h.levels_len, "levels");

        const auto* base = file_.data();
        view_.vectors = reinterpret_cast<const float*>(base + h.vectors_off);
        view_.dim = h.dim;
        view_.stride = h.stride;
        view_.count = h.count;
        view_.level0 = reinterpret_cast<const std::uint32_t*>(base + h.level0_off);
        view_.upper_off =
            reinterpret_cast<const std::uint64_t*>(base + h.upper_off_off);
        view_.upper = h.upper_len
                          ? reinterpret_cast<const std::uint32_t*>(base + h.upper_off)
                          : nullptr;
        view_.levels = reinterpret_cast<const std::int32_t*>(base + h.levels_off);
        view_.max_m0 = h.max_m0;
        view_.M = h.M;
        view_.entry = h.entry;
        view_.max_level = h.max_level;
        view_.metric = h.metric == 0 ? Metric::kL2 : Metric::kInnerProduct;

        header_ = h;
        // Search jumps around the graph; tell the kernel not to read ahead.
        file_.advise_random();
    }

    const GraphView& view() const { return view_; }
    const IndexHeader& header() const { return header_; }
    std::size_t size() const { return view_.count; }
    std::size_t dim() const { return view_.dim; }
    std::size_t bytes() const { return file_.size(); }

    std::vector<Neighbor> search(const float* q, std::size_t k, std::size_t ef,
                                 Scratch& s) const {
        return vec::search(view_, q, k, ef, s);
    }

private:
    core::MmapFile file_;
    GraphView view_;
    IndexHeader header_{};
};

}  // namespace vec
