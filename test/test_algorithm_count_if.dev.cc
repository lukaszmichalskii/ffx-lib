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
  struct count_if_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using count_if_empty_params_t = count_if_params_t<0>;
  using count_if_single_element_params_t = count_if_params_t<1>;
  using count_if_small_params_t = count_if_params_t<64>;
  using count_if_large_params_t = count_if_params_t<1024>;
  using count_if_odd_size_params_t = count_if_params_t<1007>;

  template <typename T>
  class CountIfTest : public ::testing::Test {};

  struct count_if_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, count_if_empty_params_t>)
        return "empty";
      if constexpr (std::is_same_v<T, count_if_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, count_if_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, count_if_large_params_t>)
        return "large";
      if constexpr (std::is_same_v<T, count_if_odd_size_params_t>)
        return "odd_size";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<count_if_empty_params_t,
                                  count_if_single_element_params_t,
                                  count_if_small_params_t,
                                  count_if_large_params_t,
                                  count_if_odd_size_params_t>;

  TYPED_TEST_SUITE(CountIfTest, impl_t, count_if_test_name_t);

  struct is_even_t {
    ALPAKA_FN_HOST_ACC bool operator()(const int x) const { return x % 2 == 0; }
  };

  TYPED_TEST(CountIfTest, count_even_elements) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<int> host_input(N);
    std::mt19937 gen(kSeed);
    std::uniform_int_distribution<int> dist(-500, 500);
    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
    }

    is_even_t predicate{};
    const auto expected_count = std::ranges::count_if(host_input.begin(), host_input.end(), predicate);

    if (N == 0) {
      auto device_input = ffx::make_device_buffer<int[]>(queue, 0);
      const auto result_count = ffx::algorithm::count_if(queue, device_input.data(), device_input.data(), predicate);
      EXPECT_EQ(result_count, 0);
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    const auto result_count = ffx::algorithm::count_if(queue, device_input.data(), device_input.data() + N, predicate);

    EXPECT_EQ(result_count, static_cast<decltype(result_count)>(expected_count))
        << "Element count mismatch | Got: " << result_count << " | Expected: " << expected_count;
  }

}  // namespace ffx_runtime