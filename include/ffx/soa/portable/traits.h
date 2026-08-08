#pragma once

#include "ffx/core/detail/concepts.h"
#include "ffx/soa/portable/portable_device_object.h"
#include "ffx/soa/portable/portable_device_collection.h"
#include "ffx/soa/portable/portable_host_object.h"
#include "ffx/soa/portable/portable_host_collection.h"

namespace ffx::soa::traits {

  template <concepts::device TDev, typename T>
  struct PortableObjectTrait {
    using type = PortableDeviceObject<TDev, T>;
  };

  // specialize for host device
  template <typename T>
  struct PortableObjectTrait<DevHost, T> {
    using type = PortableHostObject<T>;
  };

  // portable collection & multi-collection

  // trait for a generic SoA-based product
  template <concepts::device TDev, typename T>
  struct PortableCollectionTrait {
    using type = PortableDeviceCollection<TDev, T>;
  };

  // specialize for host device
  template <typename T>
  struct PortableCollectionTrait<DevHost, T> {
    using type = PortableHostCollection<T>;
  };

}  // namespace ffx::soa::traits