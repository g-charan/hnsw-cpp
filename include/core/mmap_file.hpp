#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace core {

// Read-only memory map, RAII and move-only.
//
// Only the read side is mapped. Writing an index happens once and is not on any
// hot path, so it uses ordinary buffered writes; reading is the path that has
// to be free, and mapping lets an index open with no parsing and no copy -- the
// pages arrive from the page cache on first touch.
class MmapFile {
public:
    MmapFile() = default;

    explicit MmapFile(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("open " + path + ": " + std::strerror(errno));
        }

        struct stat st {};
        if (::fstat(fd, &st) != 0) {
            const int e = errno;
            ::close(fd);
            throw std::runtime_error("fstat " + path + ": " + std::strerror(e));
        }
        size_ = static_cast<std::size_t>(st.st_size);

        if (size_ == 0) {
            ::close(fd);
            return;  // empty file maps to nothing; data() stays null
        }

        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        const int e = errno;
        ::close(fd);  // the mapping keeps its own reference; the fd is done
        if (p == MAP_FAILED) {
            size_ = 0;
            throw std::runtime_error("mmap " + path + ": " + std::strerror(e));
        }
        data_ = static_cast<const std::byte*>(p);
    }

    ~MmapFile() { reset(); }

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    MmapFile(MmapFile&& o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)) {}

    MmapFile& operator=(MmapFile&& o) noexcept {
        if (this != &o) {
            reset();
            data_ = std::exchange(o.data_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }

    // Hint that access will be sequential (index build) or random (search), so
    // the kernel can read ahead or refuse to.
    void advise_sequential() const {
        if (data_) ::madvise(const_cast<std::byte*>(data_), size_, MADV_SEQUENTIAL);
    }
    void advise_random() const {
        if (data_) ::madvise(const_cast<std::byte*>(data_), size_, MADV_RANDOM);
    }

    const std::byte* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }

    // Typed view at a byte offset, bounds-checked against the mapping. Every
    // read of an index file goes through here: a truncated or corrupt file must
    // fail loudly at open, not segfault somewhere inside the beam search.
    template <class T>
    const T* at(std::size_t byte_offset, std::size_t count = 1) const {
        if (byte_offset + count * sizeof(T) > size_) {
            throw std::runtime_error("mmap read out of bounds");
        }
        return reinterpret_cast<const T*>(data_ + byte_offset);
    }

private:
    void reset() {
        if (data_) {
            ::munmap(const_cast<std::byte*>(data_), size_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace core
