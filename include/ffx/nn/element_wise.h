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

  template <std::size_t Size, typename... TOperators>
  struct ElementWise {
    template <concepts::accelerator TAcc, typename... TArgs>
    ALPAKA_FN_ACC auto operator()(const TAcc& acc, TArgs... args) const -> void {
      constexpr std::size_t num_args = sizeof...(TArgs);
      auto args_tuple = std::forward_as_tuple(args...);
      auto* output = std::get<num_args - 1>(args_tuple);

      using TupleOps = std::tuple<TOperators...>;
      using FirstOp = std::tuple_element_t<0, TupleOps>;

      for (const auto thread_idx : alpaka::uniformElements(acc, Size)) {
        auto accum = [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
          return FirstOp::forward(acc, consteval_operand(std::get<Indices>(args_tuple), thread_idx)...);
        }(std::make_index_sequence<num_args - 1>{});

        [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
          ((accum = std::tuple_element_t<Indices + 1, TupleOps>::forward(acc, accum)), ...);
        }(std::make_index_sequence<sizeof...(TOperators) - 1>{});

        output[thread_idx] = accum;
      }
    }
  };

}  // namespace ffx::nn
