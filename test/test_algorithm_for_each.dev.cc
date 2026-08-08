#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  constexpr int kSeed = 42;

  template <std::size_t NumberOfElements>
  struct for_each_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using for_each_empty_params_t = for_each_params_t<0>;
  using for_each_single_element_params_t = for_each_params_t<1>;
  using for_each_small_params_t = for_each_params_t<64>;
  using for_each_large_params_t = for_each_params_t<1024>;
  using for_each_odd_size_params_t = for_each_params_t<1007>;

  template <typename T>
  class ForEachTest : public ::testing::Test {};

  struct for_each_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, for_each_empty_params_t>)
        return "empty";
      if constexpr (std::is_same_v<T, for_each_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, for_each_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, for_each_large_params_t>)
        return "large";
      if constexpr (std::is_same_v<T, for_each_odd_size_params_t>)
        return "odd_size";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<for_each_empty_params_t,
                                  for_each_single_element_params_t,
                                  for_each_small_params_t,
                                  for_each_large_params_t,
                                  for_each_odd_size_params_t>;

  TYPED_TEST_SUITE(ForEachTest, impl_t, for_each_test_name_t);

  struct double_value_t {
    ALPAKA_FN_HOST_ACC void operator()(int& x) const { x *= 2; }
  };

  TYPED_TEST(ForEachTest, double_elements) {
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

    double_value_t op{};

    std::vector<int> host_expected = host_input;
    std::ranges::for_each(host_expected.begin(), host_expected.end(), op);

    if (N == 0) {
      SUCCEED();
      return;
    }

    auto device_data = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_data, view_input);

    ffx::algorithm::for_each(queue, device_data.data(), device_data.data() + N, op);

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