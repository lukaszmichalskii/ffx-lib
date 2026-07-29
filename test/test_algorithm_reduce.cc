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
  struct reduce_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using reduce_empty_params_t = reduce_params_t<0>;
  using reduce_single_element_params_t = reduce_params_t<1>;
  using reduce_small_params_t = reduce_params_t<64>;
  using reduce_large_params_t = reduce_params_t<1024>;
  using reduce_odd_size_params_t = reduce_params_t<1007>;

  template <typename T>
  class ReduceTest : public ::testing::Test {};

  struct reduce_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, reduce_empty_params_t>)
        return "empty";
      if constexpr (std::is_same_v<T, reduce_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, reduce_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, reduce_large_params_t>)
        return "large";
      if constexpr (std::is_same_v<T, reduce_odd_size_params_t>)
        return "odd_size";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<reduce_empty_params_t,
                                  reduce_single_element_params_t,
                                  reduce_small_params_t,
                                  reduce_large_params_t,
                                  reduce_odd_size_params_t>;

  TYPED_TEST_SUITE(ReduceTest, impl_t, reduce_test_name_t);

  struct add_t {
    ALPAKA_FN_HOST_ACC int operator()(const int a, const int b) const { return a + b; }
  };

  TYPED_TEST(ReduceTest, reduce_default) {
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

    if (N == 0) {
      auto device_input = ffx::make_device_buffer<int[]>(queue, 0);
      const auto result = ffx::algorithm::reduce(queue, device_input.data(), device_input.data());
      EXPECT_EQ(result, int{});
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    const auto expected_sum = std::reduce(host_input.begin(), host_input.end());
    const auto result_sum = ffx::algorithm::reduce(queue, device_input.data(), device_input.data() + N);

    EXPECT_EQ(result_sum, expected_sum) << "Default reduce mismatch | Got: " << result_sum
                                        << " | Expected: " << expected_sum;
  }

  TYPED_TEST(ReduceTest, reduce_with_init) {
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

    constexpr int init_val = 100;

    if (N == 0) {
      auto device_input = ffx::make_device_buffer<int[]>(queue, 0);
      const auto result = ffx::algorithm::reduce(queue, device_input.data(), device_input.data(), init_val);
      EXPECT_EQ(result, init_val);
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    const auto expected_sum = std::reduce(host_input.begin(), host_input.end(), init_val);
    const auto result_sum = ffx::algorithm::reduce(queue, device_input.data(), device_input.data() + N, init_val);

    EXPECT_EQ(result_sum, expected_sum) << "Reduce with init mismatch | Got: " << result_sum
                                        << " | Expected: " << expected_sum;
  }

  TYPED_TEST(ReduceTest, reduce_with_init_and_op) {
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

    constexpr int init_val = 42;
    add_t op{};

    if (N == 0) {
      auto device_input = ffx::make_device_buffer<int[]>(queue, 0);
      const auto result = ffx::algorithm::reduce(queue, device_input.data(), device_input.data(), init_val, op);
      EXPECT_EQ(result, init_val);
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    const auto expected_sum = std::reduce(host_input.begin(), host_input.end(), init_val, op);
    const auto result_sum = ffx::algorithm::reduce(queue, device_input.data(), device_input.data() + N, init_val, op);

    EXPECT_EQ(result_sum, expected_sum) << "Reduce with custom op mismatch | Got: " << result_sum
                                        << " | Expected: " << expected_sum;
  }

}  // namespace ffx_runtime