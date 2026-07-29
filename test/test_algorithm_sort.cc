#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  constexpr int kSeed = 42;

  template <std::size_t NumberOfElements>
  struct sort_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using sort_empty_params_t = sort_params_t<0>;
  using sort_single_element_params_t = sort_params_t<1>;
  using sort_small_params_t = sort_params_t<64>;
  using sort_large_params_t = sort_params_t<1024>;
  using sort_odd_size_params_t = sort_params_t<1007>;

  template <typename T>
  class SortTest : public ::testing::Test {};

  struct sort_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, sort_empty_params_t>)
        return "empty";
      if constexpr (std::is_same_v<T, sort_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, sort_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, sort_large_params_t>)
        return "large";
      if constexpr (std::is_same_v<T, sort_odd_size_params_t>)
        return "odd_size";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<sort_empty_params_t,
                                  sort_single_element_params_t,
                                  sort_small_params_t,
                                  sort_large_params_t,
                                  sort_odd_size_params_t>;

  TYPED_TEST_SUITE(SortTest, impl_t, sort_test_name_t);

  struct greater_t {
    ALPAKA_FN_HOST_ACC bool operator()(const int a, const int b) const { return a > b; }
  };

  TYPED_TEST(SortTest, sort_default_ascending) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<int> host_input(N);
    std::mt19937 gen(kSeed);
    std::uniform_int_distribution<int> dist(-5000, 5000);
    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
    }

    std::vector<int> host_expected = host_input;
    std::ranges::sort(host_expected.begin(), host_expected.end());

    if (N == 0) {
      SUCCEED();
      return;
    }

    auto device_data = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_data, view_input);

    ffx::algorithm::sort(queue, device_data.data(), device_data.data() + N);

    auto host_output = ffx::make_host_buffer<int[]>(N);
    const auto view_device_output = alpaka::createView(device, device_data.data(), static_cast<Extent>(N));

    alpaka::memcpy(queue, host_output, view_device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_EQ(host_output[i], host_expected[i])
          << "Mismatch at index: " << i << " | Got: " << host_output[i] << " | Expected: " << host_expected[i];
    }
  }

  TYPED_TEST(SortTest, sort_custom_comparator_descending) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<int> host_input(N);
    std::mt19937 gen(kSeed + 1);
    std::uniform_int_distribution<int> dist(-5000, 5000);
    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
    }

    greater_t comp{};

    std::vector<int> host_expected = host_input;
    std::ranges::sort(host_expected.begin(), host_expected.end(), comp);

    if (N == 0) {
      SUCCEED();
      return;
    }

    auto device_data = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_data, view_input);

    ffx::algorithm::sort(queue, device_data.data(), device_data.data() + N, comp);

    auto host_output = ffx::make_host_buffer<int[]>(N);
    const auto view_device_output = alpaka::createView(device, device_data.data(), static_cast<Extent>(N));

    alpaka::memcpy(queue, host_output, view_device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_EQ(host_output[i], host_expected[i])
          << "Mismatch at index: " << i << " | Got: " << host_output[i] << " | Expected: " << host_expected[i];
    }
  }

}  // namespace ffx_runtime