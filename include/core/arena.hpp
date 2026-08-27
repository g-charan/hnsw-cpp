#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace core {

// Block-chained bump allocator.
//
// Individual allocations are never freed; the whole arena is released at once.
// This is the right shape for an HNSW graph: nodes and their neighbour arrays
// are written once during build, read many times during search, and die
// together when the index is dropped. Dropping per-object free lets an
// allocation be a pointer bump, and lets neighbour arrays for nearby nodes land
// on the same cache lines.
//
// Blocks are cache-line aligned and grow geometrically until block_size, so a
// build that allocates millions of small arrays does O(log n) real allocations.
class Arena {
public:
    static constexpr std::size_t kDefaultBlock = std::size_t{1} << 20;  // 1 MiB
    static constexpr std::size_t kBlockAlign = 64;                      // cache line

    explicit Arena(std::size_t block_size = kDefaultBlock)
        : block_size_(block_size < kBlockAlign ? kBlockAlign : block_size) {}

    ~Arena() { release(); }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    Arena(Arena&& o) noexcept
        : blocks_(std::move(o.blocks_)),
          block_size_(o.block_size_),
          used_(o.used_),
          reserved_(o.reserved_) {
        o.blocks_.clear();
        o.used_ = 0;
        o.reserved_ = 0;
    }

    Arena& operator=(Arena&& o) noexcept {
        if (this != &o) {
            release();
            blocks_ = std::move(o.blocks_);
            block_size_ = o.block_size_;
            used_ = o.used_;
            reserved_ = o.reserved_;
            o.blocks_.clear();
            o.used_ = 0;
            o.reserved_ = 0;
        }
        return *this;
    }

    // Returns memory aligned to `align`, which must be a power of two.
    void* allocate(std::size_t bytes, std::size_t align = alignof(std::max_align_t)) {
        if (bytes == 0) return nullptr;

        if (!blocks_.empty()) {
            Block& b = blocks_.back();
            std::size_t off = align_up(b.used, align);
            if (off + bytes <= b.cap) {
                void* p = b.data + off;
                b.used = off + bytes;
                used_ += bytes;
                return p;
            }
        }

        // An allocation larger than the block size gets its own exact block, so
        // one huge request cannot force every later block to be huge too.
        std::size_t cap = bytes + align > block_size_ ? bytes + align : block_size_;
        add_block(cap);

        Block& b = blocks_.back();
        std::size_t off = align_up(b.used, align);
        void* p = b.data + off;
        b.used = off + bytes;
        used_ += bytes;
        return p;
    }

    // Uninitialised storage for n objects of T. T must be trivially
    // destructible: the arena never runs destructors.
    template <class T>
    T* alloc_array(std::size_t n, std::size_t align = alignof(T)) {
        static_assert(std::is_trivially_destructible_v<T>,
                      "Arena never runs destructors");
        return static_cast<T*>(allocate(n * sizeof(T), align));
    }

    void release() {
        for (Block& b : blocks_) {
            ::operator delete(b.data, std::align_val_t{kBlockAlign});
        }
        blocks_.clear();
        used_ = 0;
        reserved_ = 0;
    }

    std::size_t bytes_used() const { return used_; }          // handed out
    std::size_t bytes_reserved() const { return reserved_; }  // actually mapped
    std::size_t block_count() const { return blocks_.size(); }

private:
    struct Block {
        std::byte* data;
        std::size_t cap;
        std::size_t used;
    };

    static std::size_t align_up(std::size_t v, std::size_t a) {
        return (v + a - 1) & ~(a - 1);
    }

    void add_block(std::size_t cap) {
        cap = align_up(cap, kBlockAlign);
        auto* mem = static_cast<std::byte*>(
            ::operator new(cap, std::align_val_t{kBlockAlign}));
        blocks_.push_back(Block{mem, cap, 0});
        reserved_ += cap;
    }

    std::vector<Block> blocks_;
    std::size_t block_size_;
    std::size_t used_ = 0;
    std::size_t reserved_ = 0;
};

}  // namespace core
