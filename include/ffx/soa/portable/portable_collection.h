#pragma once

#include "ffx/core/detail/concepts.h"
#include "ffx/soa/portable/copy_to_device.h"
#include "ffx/soa/portable/copy_to_host.h"
#include "ffx/soa/portable/portable_device_collection.h"
#include "ffx/soa/portable/portable_host_collection.h"
#include "ffx/soa/portable/traits.h"

namespace ffx::soa {

  template <concepts::device TDev, typename T>
  using PortableCollection = traits::PortableCollectionTrait<TDev, T>::type;

  namespace detail {

    template <concepts::device TDevice, typename TLayout>
    struct CopyToHost<PortableDeviceCollection<TDevice, TLayout>> {
      template <concepts::queue TQueue>
      static auto copyAsync(TQueue& queue, PortableDeviceCollection<TDevice, TLayout> const& src_data) {
        PortableHostCollection<TLayout> dst_data(src_data->metadata().size(), queue);
        alpaka::memcpy(queue, dst_data.buffer(), src_data.buffer());
        return dst_data;
      }
    };

    template <typename TLayout>
    struct CopyToDevice<PortableHostCollection<TLayout>> {
      template <concepts::non_cpu_queue TQueue>
      static auto copyAsync(TQueue& queue, PortableHostCollection<TLayout> const& src_data) {
        using TDevice = alpaka::trait::DevType<TQueue>::type;
        PortableDeviceCollection<TDevice, TLayout> dst_data(src_data->metadata().size(), queue);
        alpaka::memcpy(queue, dst_data.buffer(), src_data.buffer());
        return dst_data;
      }
    };

  }  // namespace detail

}  // namespace ffx::soa
