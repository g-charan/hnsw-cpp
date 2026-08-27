#include "check.hpp"
#include "core/arena.hpp"

#include <cstdint>
#include <vector>

int main() {
    // Alignment is the contract the vector store depends on: rows must land on
    // 16-byte boundaries for the NEON loads.
    {
        core::Arena a(4096);
        for (std::size_t align : {std::size_t{1}, std::size_t{4}, std::size_t{16},
                                  std::size_t{64}}) {
            for (int i = 0; i < 50; ++i) {
                void* p = a.allocate(17, align);
                CHECK(p != nullptr);
                CHECK(reinterpret_cast<std::uintptr_t>(p) % align == 0);
            }
        }
    }

    // Memory handed out must actually be usable and must not overlap.
    {
        core::Arena a(256);  // small blocks, so this spans many of them
        std::vector<std::uint64_t*> ptrs;
        for (std::uint64_t i = 0; i < 1000; ++i) {
            auto* p = a.alloc_array<std::uint64_t>(4);
            for (int j = 0; j < 4; ++j) p[j] = i * 10 + j;
            ptrs.push_back(p);
        }
        for (std::uint64_t i = 0; i < 1000; ++i) {
            for (int j = 0; j < 4; ++j) {
                CHECK(ptrs[i][j] == i * 10 + static_cast<std::uint64_t>(j));
            }
        }
        CHECK(a.block_count() > 1);
        CHECK(a.bytes_used() == 1000 * 4 * sizeof(std::uint64_t));
        CHECK(a.bytes_reserved() >= a.bytes_used());
    }

    // An allocation larger than the block size gets its own block rather than
    // inflating every block after it.
    {
        core::Arena a(1024);
        void* big = a.allocate(1 << 20, 64);
        CHECK(big != nullptr);
        CHECK(reinterpret_cast<std::uintptr_t>(big) % 64 == 0);
        void* small = a.allocate(8, 8);
        CHECK(small != nullptr);
        CHECK(a.bytes_reserved() < (1 << 20) + 100000);
    }

    // Move must transfer ownership without double-freeing.
    {
        core::Arena a(1024);
        auto* p = a.alloc_array<int>(10);
        p[0] = 42;
        core::Arena b(std::move(a));
        CHECK(b.bytes_used() == 10 * sizeof(int));
        CHECK(p[0] == 42);
        CHECK(a.bytes_used() == 0);
        CHECK(a.block_count() == 0);
    }

    { core::Arena a(64); CHECK(a.allocate(0) == nullptr); }

    PASS();
}
