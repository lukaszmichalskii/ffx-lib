#pragma once

#include <alpaka/alpaka.hpp>

#include "ffx/core/alpaka/config.h"

namespace ffx::detail {

  using namespace ffx_alpaka;

  template <typename TDev, typename T>
  struct buffer_type {
    using type = alpaka::Buf<TDev, T, Dim0D, Idx>;
  };

  template <typename TDev, typename T>
  struct buffer_type<TDev, T[]> {
    using type = alpaka::Buf<TDev, T, Dim1D, Idx>;
  };

  template <typename TDev, typename T, int N>
  struct buffer_type<TDev, T[N]> {
    using type = alpaka::Buf<TDev, T, Dim1D, Idx>;
  };

}  // namespace ffx::detail
