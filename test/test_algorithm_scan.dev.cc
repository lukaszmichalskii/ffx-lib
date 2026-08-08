#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  constexpr int kSeed = 42;

  template <std::size_t NumberOfElements>
  struct scan_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using scan_empty_params_t = scan_params_t<0>;
  using scan_single_element_params_t = scan_params_t<1>;
  using scan_small_params_t = scan_params_t<64>;
  using scan_large_params_t = scan_params_t<1024>;
  using scan_odd_size_params_t = scan_params_t<1007>;

  template <typename T>
  class ScanTest : public ::testing::Test {};

  struct scan_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, scan_empty_params_t>)
        return "empty";
      if constexpr (std::is_same_v<T, scan_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, scan_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, scan_large_params_t>)
        return "large";
      if constexpr (std::is_same_v<T, scan_odd_size_params_t>)
        return "odd_size";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<scan_empty_params_t,
                                  scan_single_element_params_t,
                                  scan_small_params_t,
                                  scan_large_params_t,
                                  scan_odd_size_params_t>;

  TYPED_TEST_SUITE(ScanTest, impl_t, scan_test_name_t);

  struct add_t {
    ALPAKA_FN_HOST_ACC int operator()(const int a, const int b) const { return a + b; }
  };

  TYPED_TEST(ScanTest, inclusive_scan_default) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<int> host_input(N);
    std::mt19937 gen(kSeed);
    std::uniform_int_distribution<int> dist(-50, 50);
    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
    }

    std::vector<int> host_expected(N);
    std::inclusive_scan(host_input.begin(), host_input.end(), host_expected.begin());

    if (N == 0) {
      SUCCEED();
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    auto device_output = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    ffx::algorithm::inclusive_scan(queue, device_input.data(), device_input.data() + N, device_output.data());

    auto host_output = ffx::make_host_buffer<int[]>(N);
    const auto view_device_output = alpaka::createView(device, device_output.data(), static_cast<Extent>(N));

    alpaka::memcpy(queue, host_output, view_device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_EQ(host_output[i], host_expected[i])
          << "Mismatch at index: " << i << " | Got: " << host_output[i] << " | Expected: " << host_expected[i];
    }
  }

  TYPED_TEST(ScanTest, inclusive_scan_with_init_and_op) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<int> host_input(N);
    std::mt19937 gen(kSeed + 1);
    std::uniform_int_distribution<int> dist(-50, 50);
    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
    }

    constexpr int init_val = 10;
    add_t op{};

    std::vector<int> host_expected(N);
    std::inclusive_scan(host_input.begin(), host_input.end(), host_expected.begin(), op, init_val);

    if (N == 0) {
      SUCCEED();
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    auto device_output = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    ffx::algorithm::inclusive_scan(
        queue, device_input.data(), device_input.data() + N, device_output.data(), op, init_val);

    auto host_output = ffx::make_host_buffer<int[]>(N);
    const auto view_device_output = alpaka::createView(device, device_output.data(), static_cast<Extent>(N));

    alpaka::memcpy(queue, host_output, view_device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_EQ(host_output[i], host_expected[i])
          << "Mismatch at index: " << i << " | Got: " << host_output[i] << " | Expected: " << host_expected[i];
    }
  }

  TYPED_TEST(ScanTest, exclusive_scan_with_init_and_op) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<int> host_input(N);
    std::mt19937 gen(kSeed + 2);
    std::uniform_int_distribution<int> dist(-50, 50);
    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
    }

    constexpr int init_val = 5;
    add_t op{};

    std::vector<int> host_expected(N);
    std::exclusive_scan(host_input.begin(), host_input.end(), host_expected.begin(), init_val, op);

    if (N == 0) {
      SUCCEED();
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    auto device_output = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    ffx::algorithm::exclusive_scan(
        queue, device_input.data(), device_input.data() + N, device_output.data(), init_val, op);

    auto host_output = ffx::make_host_buffer<int[]>(N);
    const auto view_device_output = alpaka::createView(device, device_output.data(), static_cast<Extent>(N));

    alpaka::memcpy(queue, host_output, view_device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_EQ(host_output[i], host_expected[i])
          << "Mismatch at index: " << i << " | Got: " << host_output[i] << " | Expected: " << host_expected[i];
    }
  }

}  // namespace ffx_runtime