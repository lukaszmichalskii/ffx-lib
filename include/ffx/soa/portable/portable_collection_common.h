#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "ffx/framework/fw_core/demangle.h"

namespace ffx::soa::detail {

  template <std::size_t I = 0, concepts::queue TQueue, typename Descriptor, typename ConstDescriptor>
    requires requires { Descriptor::num_cols; }
  void deepCopy(TQueue& queue, Descriptor& dest, ConstDescriptor const& src) {
    if constexpr (I < ConstDescriptor::num_cols) {
      assert(std::get<I>(dest.buff).size_bytes() == std::get<I>(src.buff).size_bytes());
      alpaka::memcpy(
          queue,
          alpaka::createView(alpaka::getDev(queue), std::get<I>(dest.buff).data(), std::get<I>(dest.buff).size()),
          alpaka::createView(alpaka::getDev(queue), std::get<I>(src.buff).data(), std::get<I>(src.buff).size()));
      deepCopy<I + 1>(queue, dest, src);
    }
  }

  // Helper function implementing the recursive deep copy for blocks
  template <std::size_t I = 0, concepts::queue TQueue, typename Descriptor, typename ConstDescriptor>
    requires requires { Descriptor::blocksNumber; }
  void deepCopy(TQueue& queue, Descriptor& dest, ConstDescriptor const& src) {
    if constexpr (I < ConstDescriptor::blocksNumber) {
      deepCopy(queue, std::get<I>(dest.buff), std::get<I>(src.buff));
      deepCopy<I + 1>(queue, dest, src);
    }
  }

  template <std::integral Int>
  constexpr std::size_t size_cast(Int input) {
    if constexpr (std::is_signed_v<Int>) {
      if (input < 0) {
        throw std::runtime_error(
            std::format("Invalid input for PortableCollection size: negative value {} (source type: {})",
                        input,
                        ffw::framework::type_demangle(typeid(Int).name())));
      }
    }

    if constexpr (sizeof(Int) > sizeof(std::size_t)) {
      if (std::cmp_greater(input, std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            std::format("Invalid input for PortableCollection size: value {} exceeds std::size_t max (source type: {})",
                        input,
                        ffw::framework::type_demangle(typeid(Int).name())));
      }
    }

    return static_cast<std::size_t>(input);
  }

}  // namespace ffx::soa::detail
