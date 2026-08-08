#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  constexpr int kSeed = 42;

  template <std::size_t NumberOfElements>
  struct transform_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using transform_empty_params_t = transform_params_t<0>;
  using transform_single_element_params_t = transform_params_t<1>;
  using transform_small_params_t = transform_params_t<64>;
  using transform_large_params_t = transform_params_t<1024>;
  using transform_odd_size_params_t = transform_params_t<1007>;

  template <typename T>
  class TransformTest : public ::testing::Test {};

  struct transform_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, transform_empty_params_t>)
        return "empty";
      if constexpr (std::is_same_v<T, transform_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, transform_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, transform_large_params_t>)
        return "large";
      if constexpr (std::is_same_v<T, transform_odd_size_params_t>)
        return "odd_size";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<transform_empty_params_t,
                                  transform_single_element_params_t,
                                  transform_small_params_t,
                                  transform_large_params_t,
                                  transform_odd_size_params_t>;

  TYPED_TEST_SUITE(TransformTest, impl_t, transform_test_name_t);

  struct negate_t {
    ALPAKA_FN_HOST_ACC int operator()(const int x) const { return -x; }
  };

  struct add_t {
    ALPAKA_FN_HOST_ACC int operator()(const int a, const int b) const { return a + b; }
  };

  TYPED_TEST(TransformTest, transform_unary) {
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

    negate_t op{};

    std::vector<int> host_expected(N);
    std::ranges::transform(host_input.begin(), host_input.end(), host_expected.begin(), op);

    if (N == 0) {
      auto device_input = ffx::make_device_buffer<int[]>(queue, 0);
      auto device_output = ffx::make_device_buffer<int[]>(queue, 0);
      const auto res_end =
          ffx::algorithm::transform(queue, device_input.data(), device_input.data(), device_output.data(), op);
      EXPECT_EQ(res_end, device_output.data());
      return;
    }

    auto device_input = ffx::make_device_buffer<int[]>(queue, N);
    auto device_output = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input, view_input);

    const auto device_output_end =
        ffx::algorithm::transform(queue, device_input.data(), device_input.data() + N, device_output.data(), op);

    const std::size_t processed_count = device_output_end - device_output.data();
    EXPECT_EQ(processed_count, N) << "Output iterator end offset mismatch.";

    auto host_output = ffx::make_host_buffer<int[]>(N);
    const auto view_device_output = alpaka::createView(device, device_output.data(), static_cast<Extent>(N));

    alpaka::memcpy(queue, host_output, view_device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_EQ(host_output[i], host_expected[i])
          << "Unary transform mismatch at index: " << i << " | Got: " << host_output[i]
          << " | Expected: " << host_expected[i];
    }
  }

  TYPED_TEST(TransformTest, transform_binary) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<int> host_input1(N);
    std::vector<int> host_input2(N);
    std::mt19937 gen(kSeed + 1);
    std::uniform_int_distribution<int> dist(-5000, 5000);
    for (std::size_t i = 0; i < N; ++i) {
      host_input1[i] = dist(gen);
      host_input2[i] = dist(gen);
    }

    add_t op{};

    std::vector<int> host_expected(N);
    std::transform(host_input1.begin(), host_input1.end(), host_input2.begin(), host_expected.begin(), op);

    if (N == 0) {
      auto device_input1 = ffx::make_device_buffer<int[]>(queue, 0);
      auto device_input2 = ffx::make_device_buffer<int[]>(queue, 0);
      auto device_output = ffx::make_device_buffer<int[]>(queue, 0);
      const auto res_end = ffx::algorithm::transform(
          queue, device_input1.data(), device_input1.data(), device_input2.data(), device_output.data(), op);
      EXPECT_EQ(res_end, device_output.data());
      return;
    }

    auto device_input1 = ffx::make_device_buffer<int[]>(queue, N);
    auto device_input2 = ffx::make_device_buffer<int[]>(queue, N);
    auto device_output = ffx::make_device_buffer<int[]>(queue, N);

    const auto view_input1 = alpaka::createView(ffx::host(), host_input1.data(), static_cast<Extent>(N));
    const auto view_input2 = alpaka::createView(ffx::host(), host_input2.data(), static_cast<Extent>(N));
    alpaka::memcpy(queue, device_input1, view_input1);
    alpaka::memcpy(queue, device_input2, view_input2);

    const auto device_output_end = ffx::algorithm::transform(
        queue, device_input1.data(), device_input1.data() + N, device_input2.data(), device_output.data(), op);

    const std::size_t processed_count = device_output_end - device_output.data();
    EXPECT_EQ(processed_count, N) << "Output iterator end offset mismatch.";

    auto host_output = ffx::make_host_buffer<int[]>(N);
    const auto view_device_output = alpaka::createView(device, device_output.data(), static_cast<Extent>(N));

    alpaka::memcpy(queue, host_output, view_device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_EQ(host_output[i], host_expected[i])
          << "Binary transform mismatch at index: " << i << " | Got: " << host_output[i]
          << " | Expected: " << host_expected[i];
    }
  }

}  // namespace ffx_runtime