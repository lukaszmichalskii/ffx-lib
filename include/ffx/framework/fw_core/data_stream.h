#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "ffx/framework/data_formats/batch.h"
#include "ffx/framework/data_formats/header.h"

namespace ffx::framework {

  template <typename T>
  class DataStream {
  public:
    explicit DataStream(const std::string_view filepath) {
      file_descriptor_ = open(filepath.data(), O_RDONLY);
      if (file_descriptor_ < 0)
        throw std::runtime_error("Failed to open binary stream file");

      struct stat st {};
      fstat(file_descriptor_, &st);
      size_ = static_cast<std::size_t>(st.st_size);

      void* mapped = mmap(nullptr, size_, PROT_READ, MAP_SHARED, file_descriptor_, 0);
      if (mapped == MAP_FAILED) {
        close(file_descriptor_);
        throw std::runtime_error("mmap failed");
      }

      byte_data_ = static_cast<const uint8_t*>(mapped);
      byte_ptr_ = byte_data_;
      byte_ptr_end_ = byte_data_ + size_;

      (void)madvise(const_cast<uint8_t*>(byte_data_), size_, MADV_SEQUENTIAL);
    }

    ~DataStream() {
      if (byte_data_ && byte_data_ != MAP_FAILED)
        munmap(const_cast<void*>(static_cast<const void*>(byte_data_)), size_);
      if (file_descriptor_ >= 0)
        close(file_descriptor_);
    }

    // Non-copyable: owns the mmap region and file descriptor.
    DataStream(const DataStream&) = delete;
    DataStream& operator=(const DataStream&) = delete;

    DataStream(DataStream&& other) noexcept
        : file_descriptor_(other.file_descriptor_),
          size_(other.size_),
          byte_data_(other.byte_data_),
          byte_ptr_(other.byte_ptr_),
          byte_ptr_end_(other.byte_ptr_end_) {
      other.file_descriptor_ = -1;
      other.byte_data_ = nullptr;
    }

    DataStream& operator=(DataStream&&) = delete;

    std::optional<batch_t<T>> get() {
      static_assert(std::is_trivially_copyable_v<T>, "Stream types must be trivially copyable for zero-copy DMA!");

      while (byte_ptr_ < byte_ptr_end_) {
        if (byte_ptr_ + sizeof(header_t) > byte_ptr_end_) {
          byte_ptr_ = byte_ptr_end_;
          break;
        }

        header_t header{};
        std::memcpy(&header, byte_ptr_, sizeof(header_t));

        const uint8_t* payload_start = byte_ptr_ + sizeof(header_t);
        const auto payload_bytes = header.batch_size() * sizeof(T);

        if (payload_start + payload_bytes > byte_ptr_end_) {
          byte_ptr_ = byte_ptr_end_;
          break;
        }

        if (header.error_bit()) {
          byte_ptr_ = payload_start + payload_bytes;
          continue;
        }

        byte_ptr_ = payload_start + payload_bytes;
        return batch_t<T>{header, reinterpret_cast<const T*>(payload_start), header.batch_size()};
      }

      return std::nullopt;
    }

  private:
    int file_descriptor_{-1};
    std::size_t size_{0};
    const uint8_t* byte_data_{nullptr};
    const uint8_t* byte_ptr_{nullptr};
    const uint8_t* byte_ptr_end_{nullptr};
  };

}  // namespace ffx::framework
