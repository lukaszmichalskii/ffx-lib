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
  struct copy_if_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using copy_if_empty_params_t = copy_if_params_t<0>;
  using copy_if_single_element_params_t = copy_if_params_t<1>;
  using copy_if_small_params_t = copy_if_params_t<64>;
  using copy_if_large_params_t = copy_if_params_t<1024>;
  using copy_if_odd_size_params_t = copy_if_params_t<1007>;

  template <typename T>
  class CopyIfTest : public ::testing::Test {};

  struct copy_if_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, copy_if_empty_params_t>)
        return "empty";
      if constexpr (std::is_same_v<T, copy_if_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, copy_if_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, copy_if_large_params_t>)
        return "large";
      if constexpr (std::is_same_v<T, copy_if_odd_size_params_t>)
        return "odd_size";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<copy_if_empty_params_t,
                                  copy_if_single_element_params_t,
                                  copy_if_small_params_t,
                                  copy_if_large_params_t,
                                  copy_if_odd_size_params_t>;

  TYPED_TEST_SUITE(CopyIfTest, impl_t, copy_if_test_name_t);

  struct is_even_t {
    ALPAKA_FN_HOST_ACC bool operator()(const int x) const { return x % 2 == 0; }
  };

  TYPED_TEST(CopyIfTest, filter_even_elements) {
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

    std::vector<int> host_expected;
    is_even_t predicate{};
    std::ranges::copy_if(host_input.begin(), host_input.end(), std::back_inserter(host_expected), predicate);

    if (N == 0) {
      SUCCEED();
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    auto device_output = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    auto* device_output_end =
        ffx::algorithm::copy_if(queue, device_input.data(), device_input.data() + N, device_output.data(), predicate);

    const std::size_t copied_count = device_output_end - device_output.data();

    ASSERT_EQ(copied_count, host_expected.size()) << "Stream compaction output size mismatch.";

    if (copied_count > 0) {
      auto host_output = ffx::make_host_buffer<int[]>(copied_count);
      const auto view_device_output =
          alpaka::createView(device, device_output.data(), static_cast<Extent>(copied_count));

      alpaka::memcpy(queue, host_output, view_device_output);
      alpaka::wait(queue);

      for (std::size_t i = 0; i < copied_count; ++i) {
        EXPECT_EQ(host_output[i], host_expected[i])
            << "Mismatch at output index: " << i << " | Got: " << host_output[i] << " | Expected: " << host_expected[i];
      }
    }
  }

}  // namespace ffx_runtime
