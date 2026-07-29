#pragma once

#include "ffx/core/alpaka/config.h"
#include "ffx/core/alpaka/host.h"
#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/view/traits.h"

namespace ffx {

  using namespace ffx_alpaka;

  template <typename T>
  using host_view = detail::view_type<DevHost, T>::type;

  template <concepts::scalar T>
  host_view<T> make_host_view(T& data) {
    return alpaka::ViewPlainPtr<DevHost, T, Dim0D, Idx>(&data, host(), Scalar{});
  }

  template <concepts::scalar T>
  host_view<T[]> make_host_view(T* data, Extent extent) {
    return alpaka::ViewPlainPtr<DevHost, T, Dim1D, Idx>(data, host(), Vec1D{extent});
  }

  template <concepts::unbounded_array T>
  host_view<T> make_host_view(T& data, Extent extent) {
    return alpaka::ViewPlainPtr<DevHost, std::remove_extent_t<T>, Dim1D, Idx>(data, host(), Vec1D{extent});
  }

  template <concepts::bounded_array T>
  host_view<T> make_host_view(T& data) {
    return alpaka::ViewPlainPtr<DevHost, std::remove_extent_t<T>, Dim1D, Idx>(data, host(), Vec1D{std::extent_v<T>});
  }

  template <typename T>
  host_view<T[]> make_host_view(std::span<T> span) {
    return alpaka::ViewPlainPtr<DevHost, T, Dim1D, Idx>(span.data(), host(), Vec1D{span.size()});
  }

  template <typename T>
  host_view<T[]> make_host_view(std::span<T> span, Extent extent) {
    if (extent > span.size())
      throw std::runtime_error("make_host_view: span size is smaller than the specified extent");
    return alpaka::ViewPlainPtr<DevHost, T, Dim1D, Idx>(span.data(), host(), Vec1D{extent});
  }

}  // namespace ffx
