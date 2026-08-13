#pragma once

#include <cstdint>

namespace ffx::framework {

  // Universal 64-Bit Stream Control Word (8 Bytes)
  // Bits  0..10 (10b) : batch_id     -> 0 ... 1023
  // Bits 10..17 (8b)  : batch_size   -> 0 ... 255
  // Bits 18..62 (45b) : user_data    -> custom user metadata / timestamps / stream IDs
  // Bit  63     (1b)  : error_flag   -> corrupted payload skip indicator
#pragma pack(push, 1)
  struct alignas(8) header_t {
    uint64_t raw{0};

    // Bit Extractors
    constexpr uint16_t batch_id() const noexcept { return static_cast<uint16_t>(raw & 0x3FF); }
    constexpr uint8_t batch_size() const noexcept { return static_cast<uint8_t>((raw >> 10) & 0xFF); }
    constexpr uint64_t user_data() const noexcept { return (raw >> 18) & 0x1FFFFFFFFFFFULL; }
    constexpr bool error_bit() const noexcept { return ((raw >> 63) & 0x01) != 0; }

    // Constructor
    constexpr header_t(const uint16_t batch_id = 0,
                       const uint8_t batch_size = 0,
                       const uint64_t user_data = 0,
                       const bool error_bit = false) noexcept
        : raw((static_cast<uint64_t>(batch_id) & 0x3FFULL) | ((static_cast<uint64_t>(batch_size) & 0xFFULL) << 10) |
              ((static_cast<uint64_t>(user_data) & 0x1FFFFFFFFFFFULL) << 18) |
              (static_cast<uint64_t>(error_bit) << 63)) {}
  };
#pragma pack(pop)

}  // namespace ffx::framework
