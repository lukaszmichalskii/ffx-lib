#include <gtest/gtest.h>
#include <random>
#include <algorithm>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  // ---------------------------------------------------------------------------
  // Helper: run ReLU6<N> on a host vector and return the host output vector
  // ---------------------------------------------------------------------------
  template <std::size_t N>
  static std::vector<float> run_relu6(const std::vector<float>& host_input) {
    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto input_view = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    auto host_output = ffx::make_host_buffer<float[]>(N);
    auto device_input = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t tpb = 64u;
    const auto grid = ffx::make_workdiv<Acc1D>(ffx::divide_up_by(N, tpb), tpb);
    alpaka::exec<Acc1D>(queue, grid, ffx::nn::ReLU6<N>{}, device_input.data(), device_output.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    return std::vector<float>(host_output.data(), host_output.data() + N);
  }

  // =========================================================================
  // Typed test: numerical accuracy across multiple tensor sizes with random input
  // Distribution [-8, 8] guarantees elements below 0, in (0,6) and above 6
  // =========================================================================
  template <std::size_t NumberOfElements>
  struct relu6_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using relu6_small_params_t = relu6_params_t<16>;
  using relu6_matrix_params_t = relu6_params_t<8 * 32>;
  using relu6_single_element_params_t = relu6_params_t<1>;
  using relu6_large_bulk_params_t = relu6_params_t<1024>;

  template <typename T>
  class ReLU6Test : public ::testing::Test {};

  struct relu6_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, relu6_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, relu6_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, relu6_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, relu6_large_bulk_params_t>)
        return "large_bulk";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::
      Types<relu6_small_params_t, relu6_matrix_params_t, relu6_single_element_params_t, relu6_large_bulk_params_t>;
  TYPED_TEST_SUITE(ReLU6Test, impl_t, relu6_test_name_t);

  TYPED_TEST(ReLU6Test, forward_random) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    std::vector<float> host_input(N);
    std::vector<float> host_expected(N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);

    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
      host_expected[i] = std::min(6.0f, std::max(0.0f, host_input[i]));
    }

    const auto output = run_relu6<N>(host_input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(output[i], host_expected[i], 1e-6f) << "Mismatch at index " << i << " | input=" << host_input[i];
    }
  }

  // =========================================================================
  // Property tests — boundary / mathematical guarantees specific to ReLU6
  // =========================================================================

  // All negative inputs → clamped to 0 (lower bound)
  TEST(ReLU6PropertyTest, all_negative_inputs_produce_zero) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-100.0f, -1e-4f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_relu6<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(output[i], 0.0f) << "Expected 0 for negative input=" << input[i];
  }

  // Linear region (0, 6) — output must equal input exactly (identity)
  TEST(ReLU6PropertyTest, linear_region_is_identity) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(13);
    std::uniform_real_distribution<float> dist(1e-4f, 5.9999f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_relu6<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], input[i], 1e-6f) << "Identity region broken at input=" << input[i];
  }

  // All inputs above 6 → clamped to 6 (upper bound)
  TEST(ReLU6PropertyTest, all_above_cap_produce_six) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(17);
    std::uniform_real_distribution<float> dist(6.0001f, 100.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_relu6<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], 6.0f, 1e-6f) << "Expected 6 for input=" << input[i];
  }

  // Exact boundary values must pass through unmodified
  TEST(ReLU6PropertyTest, exact_boundaries_zero_and_six) {
    constexpr std::size_t N = 2;
    const std::vector<float> input = {0.0f, 6.0f};
    const auto output = run_relu6<N>(input);
    EXPECT_EQ(output[0], 0.0f) << "x=0 boundary";
    EXPECT_EQ(output[1], 6.0f) << "x=6 boundary";
  }

  // Output is always bounded in [0, 6] — the fundamental ReLU6 invariant
  TEST(ReLU6PropertyTest, output_always_in_valid_range) {
    constexpr std::size_t N = 512;
    std::vector<float> input(N);
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_relu6<N>(input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_GE(output[i], 0.0f) << "Output below 0 at index " << i;
      EXPECT_LE(output[i], 6.0f) << "Output above 6 at index " << i;
    }
  }

  // All-zero input → all-zero output
  TEST(ReLU6PropertyTest, zero_input_produces_zero_output) {
    constexpr std::size_t N = 32;
    const std::vector<float> input(N, 0.0f);
    const auto output = run_relu6<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(output[i], 0.0f) << "Non-zero output for zero input at " << i;
  }

}  // namespace ffx_runtime
