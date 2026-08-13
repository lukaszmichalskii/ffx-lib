#pragma once

#include <cstddef>
#include <thread>

namespace ffx::framework::concurrency {

  inline std::size_t get_number_of_threads(const std::size_t num_of_threads = std::thread::hardware_concurrency()) {
    const auto max_number_of_threads = static_cast<std::size_t>(std::thread::hardware_concurrency());
    const auto number_of_threads = std::max(1zu, std::min(num_of_threads, max_number_of_threads));
    return number_of_threads;
  }

}  // namespace ffx::framework::concurrency
