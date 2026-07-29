#pragma once

#include <type_traits>

#include <alpaka/alpaka.hpp>

namespace ffx::concepts {

  template <typename T>
  concept queue = alpaka::isQueue<T>;

  template <typename T>
  concept non_cpu_queue = alpaka::isQueue<T> and not std::is_same_v<alpaka::Dev<T>, alpaka::DevCpu>;

  template <typename T>
  concept device = alpaka::isDevice<T>;

  template <typename T>
  concept accelerator = alpaka::concepts::Acc<T>;

  template <typename T>
  concept platform = alpaka::isPlatform<T>;

  template <typename T>
  concept bounded_array = std::is_bounded_array_v<T> && not std::is_array_v<std::remove_extent_t<T>>;

  template <typename T>
  concept unbounded_array = std::is_unbounded_array_v<T> && not std::is_array_v<std::remove_extent_t<T>>;

  template <typename T>
  concept scalar = not std::is_array_v<T>;

  template <typename T>
  concept numeric = requires {
    std::is_arithmetic_v<T>;
    requires sizeof(T) <= 8;
  };

}  // namespace ffx::concepts
