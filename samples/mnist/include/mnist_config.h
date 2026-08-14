#pragma once

namespace ffx_mnist::config {

  constexpr auto kBatchSize = 64zu;
  constexpr auto kNumberOfClasses = 10zu;
  constexpr auto kThreadsPerBlock = 64zu;

  constexpr auto kNumberOfThreads = 8zu;
  constexpr auto kConcurrentLaneCapacity = 2zu;
  constexpr std::string_view kFilepath = "data.raw";

  struct mnist_sample_t {
    uint32_t label{0};
    std::array<uint8_t, 28 * 28> pixels{};
  };

}  // namespace ffx_mnist::config
