#pragma once

#include "ffx/core/detail/concepts.h"
#include "ffx/soa/portable/portable_device_object.h"
#include "ffx/soa/portable/portable_host_object.h"
#include "ffx/soa/portable/copy_to_device.h"
#include "ffx/soa/portable/copy_to_host.h"
#include "ffx/soa/portable/traits.h"

namespace ffx::soa {

  // type alias for a generic struct-based product
  template <concepts::device TDev, typename T>
  using PortableObject = traits::PortableObjectTrait<TDev, T>::type;

  namespace detail {

    template <typename T, typename TDevice>
    struct CopyToHost<PortableDeviceObject<T, TDevice>> {
      template <concepts::queue TQueue>
      static auto copyAsync(TQueue& queue, PortableDeviceObject<T, TDevice> const& src_data) {
        PortableHostObject<T> dest_data(queue);
        alpaka::memcpy(queue, dest_data.buffer(), src_data.buffer());
        return dest_data;
      }
    };

    template <typename T>
    struct CopyToDevice<PortableHostObject<T>> {
      template <concepts::non_cpu_queue TQueue>
      static auto copyAsync(TQueue& queue, PortableHostObject<T> const& src_data) {
        using TDevice = alpaka::trait::DevType<TQueue>::type;
        PortableDeviceObject<T, TDevice> dest_data(queue);
        alpaka::memcpy(queue, dest_data.buffer(), src_data.buffer());
        return dest_data;
      }
    };

  }  // namespace detail

}  // namespace ffx::soa
