#pragma once

#include <alpaka/alpaka.hpp>
#include <tuple>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn {

  template <typename T>
  ALPAKA_FN_ACC constexpr decltype(auto) consteval_operand(T&& arg, std::size_t idx) noexcept {
    if constexpr (std::is_pointer_v<std::remove_cvref_t<T>>) {
      return arg[idx];
    } else {
      return arg;
    }
  }

  template <std::size_t Size, typename TOperator>
  struct ElementWise {
    template <concepts::accelerator TAcc, typename... TArgs>
    ALPAKA_FN_ACC auto operator()(const TAcc& acc, TArgs... args) const -> void {
      constexpr std::size_t num_args = sizeof...(TArgs);
      auto args_tuple = std::forward_as_tuple(args...);
      auto* output = std::get<num_args - 1>(args_tuple);

      for (const auto thread_idx : alpaka::uniformElements(acc, Size)) {
        // helper lambda to index individual input buffer
        output[thread_idx] = [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
          return TOperator::forward(acc, consteval_operand(std::get<Indices>(args_tuple), thread_idx)...);
        }(std::make_index_sequence<num_args - 1>{});
      }
    }
  };

}  // namespace ffx::nn
