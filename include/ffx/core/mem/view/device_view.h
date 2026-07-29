#pragma once

#include "ffx/core/alpaka/config.h"
#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/view/traits.h"

#include <span>

namespace ffx {

  using namespace ffx_alpaka;

  template <typename TDev, typename T>
  using device_view = detail::view_type<TDev, T>::type;

  template <concepts::scalar T, concepts::device TDev>
  device_view<TDev, T> make_device_view(TDev const& device, T& data) {
    return alpaka::ViewPlainPtr<TDev, T, Dim0D, Idx>(&data, device, Scalar{});
  }

  template <concepts::scalar T, concepts::device TDev>
  device_view<TDev, T[]> make_device_view(TDev const& device, T* data, Extent extent) {
    return alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>(data, device, Vec1D{extent});
  }

  template <concepts::bounded_array T, concepts::device TDev>
  device_view<TDev, T> make_device_view(TDev const& device, T& data) {
    return alpaka::ViewPlainPtr<TDev, std::remove_extent_t<T>, Dim1D, Idx>(data, device, Vec1D{std::extent_v<T>});
  }

  template <concepts::unbounded_array T, concepts::device TDev>
  device_view<TDev, T> make_device_view(TDev const& device, T& data, Extent extent) {
    return alpaka::ViewPlainPtr<TDev, std::remove_extent_t<T>, Dim1D, Idx>(data, device, Vec1D{extent});
  }

  template <typename T, concepts::device TDev>
  device_view<TDev, T[]> make_device_view(TDev const& device, std::span<T> span) {
    return alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>(span.data(), device, Vec1D{span.size()});
  }

  template <typename T, concepts::device TDev>
  device_view<TDev, T[]> make_device_view(TDev const& device, std::span<T> span, Extent extent) {
    if (extent > span.size())
      throw std::runtime_error("make_device_view: span size is smaller than the specified extent");
    return alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>(span.data(), device, Vec1D{extent});
  }

  template <concepts::scalar T, concepts::queue TQueue>
  device_view<alpaka::Dev<TQueue>, T> make_device_view(TQueue const& queue, T& data) {
    return alpaka::ViewPlainPtr<alpaka::Dev<TQueue>, T, Dim0D, Idx>(&data, alpaka::getDev(queue), Scalar{});
  }

  template <concepts::scalar T, concepts::queue TQueue>
  device_view<alpaka::Dev<TQueue>, T[]> make_device_view(TQueue const& queue, T* data, Extent extent) {
    return alpaka::ViewPlainPtr<alpaka::Dev<TQueue>, T, Dim1D, Idx>(data, alpaka::getDev(queue), Vec1D{extent});
  }

  template <concepts::unbounded_array T, concepts::queue TQueue>
  device_view<alpaka::Dev<TQueue>, T> make_device_view(TQueue const& queue, T& data, Extent extent) {
    return alpaka::ViewPlainPtr<alpaka::Dev<TQueue>, std::remove_extent_t<T>, Dim1D, Idx>(
        data, alpaka::getDev(queue), Vec1D{extent});
  }

  template <concepts::bounded_array T, concepts::queue TQueue>
  device_view<alpaka::Dev<TQueue>, T> make_device_view(TQueue const& queue, T& data) {
    return alpaka::ViewPlainPtr<alpaka::Dev<TQueue>, std::remove_extent_t<T>, Dim1D, Idx>(
        data, alpaka::getDev(queue), Vec1D{std::extent_v<T>});
  }

  template <typename T, concepts::queue TQueue>
  device_view<alpaka::Dev<TQueue>, T[]> make_device_view(TQueue const& queue, std::span<T> span) {
    return alpaka::ViewPlainPtr<alpaka::Dev<TQueue>, T, Dim1D, Idx>(
        span.data(), alpaka::getDev(queue), Vec1D{span.size()});
  }

  template <typename T, concepts::queue TQueue>
  device_view<alpaka::Dev<TQueue>, T[]> make_device_view(TQueue const& queue, std::span<T> span, Extent extent) {
    if (extent > span.size())
      throw std::runtime_error("make_device_view: span size is smaller than the specified extent");
    return alpaka::ViewPlainPtr<alpaka::Dev<TQueue>, T, Dim1D, Idx>(span.data(), alpaka::getDev(queue), Vec1D{extent});
  }

}  // namespace ffx
