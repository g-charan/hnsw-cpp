#pragma once

#include "core/arena.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace vec {

// Contiguous float32 vector storage, one arena block, fixed stride.
//
// Rows are padded to a multiple of four floats so every row starts 16-byte
// aligned and the NEON loads never straddle. The padding is zeroed, which also
// means a kernel run over the full stride would still give the right answer --
// the kernels take `dim` anyway, but the invariant costs nothing to hold.
class VectorStore {
public:
    VectorStore(std::size_t dim, std::size_t capacity)
        : arena_(1u << 22), dim_(dim), stride_((dim + 3) & ~std::size_t{3}),
          capacity_(capacity) {
        if (dim == 0) throw std::invalid_argument("VectorStore: dim must be > 0");
        data_ = arena_.alloc_array<float>(stride_ * capacity, 64);
        std::memset(data_, 0, stride_ * capacity * sizeof(float));
    }

    std::size_t add(const float* v) {
        if (size_ >= capacity_) throw std::runtime_error("VectorStore full");
        std::memcpy(data_ + size_ * stride_, v, dim_ * sizeof(float));
        return size_++;
    }

    const float* get(std::size_t id) const { return data_ + id * stride_; }
    float* mutable_get(std::size_t id) { return data_ + id * stride_; }

    std::size_t dim() const { return dim_; }
    std::size_t stride() const { return stride_; }
    std::size_t size() const { return size_; }
    std::size_t capacity() const { return capacity_; }
    std::size_t bytes() const { return stride_ * capacity_ * sizeof(float); }

    // Bulk load: the common case is a whole dataset arriving at once.
    void add_many(const float* vs, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) add(vs + i * dim_);
    }

private:
    core::Arena arena_;
    float* data_ = nullptr;
    std::size_t dim_;
    std::size_t stride_;
    std::size_t size_ = 0;
    std::size_t capacity_;
};

}  // namespace vec
