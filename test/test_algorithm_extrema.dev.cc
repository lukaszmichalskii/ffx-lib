#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  constexpr int kSeed = 42;

  template <std::size_t NumberOfElements>
  struct extrema_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using extrema_empty_params_t = extrema_params_t<0>;
  using extrema_single_element_params_t = extrema_params_t<1>;
  using extrema_small_params_t = extrema_params_t<64>;
  using extrema_large_params_t = extrema_params_t<1024>;
  using extrema_odd_size_params_t = extrema_params_t<1007>;

  template <typename T>
  class ExtremaTest : public ::testing::Test {};

  struct extrema_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, extrema_empty_params_t>)
        return "empty";
      if constexpr (std::is_same_v<T, extrema_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, extrema_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, extrema_large_params_t>)
        return "large";
      if constexpr (std::is_same_v<T, extrema_odd_size_params_t>)
        return "odd_size";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<extrema_empty_params_t,
                                  extrema_single_element_params_t,
                                  extrema_small_params_t,
                                  extrema_large_params_t,
                                  extrema_odd_size_params_t>;

  TYPED_TEST_SUITE(ExtremaTest, impl_t, extrema_test_name_t);

  struct custom_less_t {
    ALPAKA_FN_HOST_ACC bool operator()(const int a, const int b) const { return a < b; }
  };

  TYPED_TEST(ExtremaTest, min_element_default) {
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

    if (N == 0) {
      auto device_input = ffx::make_device_buffer<int[]>(queue, 0);
      auto res = ffx::algorithm::min_element(queue, device_input.data(), device_input.data());
      EXPECT_EQ(res, device_input.data());
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    const auto expected_it = std::ranges::min_element(host_input.begin(), host_input.end());
    const std::size_t expected_idx = std::distance(host_input.begin(), expected_it);

    const auto device_min_it = ffx::algorithm::min_element(queue, device_input.data(), device_input.data() + N);

    const std::size_t result_idx = device_min_it - device_input.data();

    ASSERT_LT(result_idx, N);
    EXPECT_EQ(host_input[result_idx], *expected_it) << "Value mismatch for min_element default.";
    EXPECT_EQ(result_idx, expected_idx) << "Index mismatch for min_element default.";
  }

  TYPED_TEST(ExtremaTest, max_element_custom_predicate) {
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

    custom_less_t comp{};

    if (N == 0) {
      auto device_input = ffx::make_device_buffer<int[]>(queue, 0);
      auto res = ffx::algorithm::max_element(queue, device_input.data(), device_input.data(), comp);
      EXPECT_EQ(res, device_input.data());
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    const auto expected_it = std::ranges::max_element(host_input.begin(), host_input.end(), comp);
    const std::size_t expected_idx = std::distance(host_input.begin(), expected_it);

    const auto device_max_it = ffx::algorithm::max_element(queue, device_input.data(), device_input.data() + N, comp);

    const std::size_t result_idx = device_max_it - device_input.data();

    ASSERT_LT(result_idx, N);
    EXPECT_EQ(host_input[result_idx], *expected_it) << "Value mismatch for max_element with predicate.";
    EXPECT_EQ(result_idx, expected_idx) << "Index mismatch for max_element with predicate.";
  }

  TYPED_TEST(ExtremaTest, minmax_element_default) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<int> host_input(N);
    std::mt19937 gen(kSeed + 2);
    std::uniform_int_distribution<int> dist(-5000, 5000);
    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
    }

    if (N == 0) {
      auto device_input = ffx::make_device_buffer<int[]>(queue, 0);
      auto res = ffx::algorithm::minmax_element(queue, device_input.data(), device_input.data());
      EXPECT_EQ(res.first, device_input.data());
      EXPECT_EQ(res.second, device_input.data());
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    const auto expected_pair = std::minmax_element(host_input.begin(), host_input.end());
    const std::size_t expected_min_idx = std::distance(host_input.begin(), expected_pair.first);
    const std::size_t expected_max_idx = std::distance(host_input.begin(), expected_pair.second);

    const auto device_res = ffx::algorithm::minmax_element(queue, device_input.data(), device_input.data() + N);

    const std::size_t result_min_idx = device_res.first - device_input.data();
    const std::size_t result_max_idx = device_res.second - device_input.data();

    ASSERT_LT(result_min_idx, N);
    ASSERT_LT(result_max_idx, N);

    EXPECT_EQ(host_input[result_min_idx], *expected_pair.first) << "Min value mismatch in minmax_element.";
    EXPECT_EQ(host_input[result_max_idx], *expected_pair.second) << "Max value mismatch in minmax_element.";

    EXPECT_EQ(result_min_idx, expected_min_idx) << "Min index mismatch in minmax_element.";
    EXPECT_EQ(result_max_idx, expected_max_idx) << "Max index mismatch in minmax_element.";
  }

}  // namespace ffx_runtime