#pragma once

#include "ffx/framework/data_formats/header.h"

namespace ffx::framework {

  template <typename T>
  struct batch_t {
    header_t header{};
    const T* data{nullptr};
    std::size_t size{0};
  };

}  // namespace ffx::framework