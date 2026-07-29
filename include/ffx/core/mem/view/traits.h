#pragma once

#include <alpaka/alpaka.hpp>

#include "ffx/core/alpaka/config.h"

namespace ffx::detail {

  using namespace ffx_alpaka;

  template <typename TDev, typename T>
  struct view_type {
    using type = alpaka::ViewPlainPtr<TDev, T, Dim0D, Idx>;
  };

  template <typename TDev, typename T>
  struct view_type<TDev, T[]> {
    using type = alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>;
  };

  template <typename TDev, typename T, int N>
  struct view_type<TDev, T[N]> {
    using type = alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>;
  };

}  // namespace ffx::detail
