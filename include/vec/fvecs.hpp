#pragma once

#include "core/mmap_file.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace vec {

// Readers for the .fvecs/.ivecs formats the ANN benchmark datasets ship in
// (SIFT1M, GIST1M, from INRIA corpus-texmex).
//
// The format is as blunt as it sounds: each record is a little-endian int32
// dimension followed by that many float32 (or int32) values, repeated to EOF.
// There is no header and no count, so the record count is derived from the file
// size -- which doubles as a corruption check, since a truncated file will not
// divide evenly.

template <class T>
struct VecFile {
    std::size_t dim = 0;
    std::size_t count = 0;
    std::vector<T> data;  // count * dim, row-major, dimension prefixes stripped

    const T* row(std::size_t i) const { return data.data() + i * dim; }
};

template <class T>
VecFile<T> read_xvecs(const std::string& path) {
    static_assert(sizeof(T) == 4, "xvecs elements are 4 bytes");

    core::MmapFile f(path);
    f.advise_sequential();
    if (!f.valid() || f.size() < 4) {
        throw std::runtime_error("xvecs: empty or too small: " + path);
    }

    std::int32_t dim0 = 0;
    std::memcpy(&dim0, f.data(), 4);
    if (dim0 <= 0) {
        throw std::runtime_error("xvecs: bad leading dimension in " + path);
    }

    const auto dim = static_cast<std::size_t>(dim0);
    const std::size_t record = 4 + dim * sizeof(T);
    if (f.size() % record != 0) {
        throw std::runtime_error("xvecs: size " + std::to_string(f.size()) +
                                 " is not a multiple of record size " +
                                 std::to_string(record) + " -- truncated or wrong dim: " +
                                 path);
    }

    VecFile<T> out;
    out.dim = dim;
    out.count = f.size() / record;
    out.data.resize(out.count * dim);

    for (std::size_t i = 0; i < out.count; ++i) {
        const std::byte* rec = f.data() + i * record;
        std::int32_t d = 0;
        std::memcpy(&d, rec, 4);
        if (static_cast<std::size_t>(d) != dim) {
            throw std::runtime_error("xvecs: ragged dimensions at record " +
                                     std::to_string(i) + " in " + path);
        }
        std::memcpy(out.data.data() + i * dim, rec + 4, dim * sizeof(T));
    }
    return out;
}

inline VecFile<float> read_fvecs(const std::string& path) {
    return read_xvecs<float>(path);
}
inline VecFile<std::int32_t> read_ivecs(const std::string& path) {
    return read_xvecs<std::int32_t>(path);
}

// Writer, used by the tests to round-trip and by tools that emit synthetic sets.
template <class T>
void write_xvecs(const std::string& path, const T* data, std::size_t count,
                 std::size_t dim) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    const auto d32 = static_cast<std::int32_t>(dim);
    for (std::size_t i = 0; i < count; ++i) {
        std::fwrite(&d32, 4, 1, f);
        std::fwrite(data + i * dim, sizeof(T), dim, f);
    }
    std::fclose(f);
}

}  // namespace vec
