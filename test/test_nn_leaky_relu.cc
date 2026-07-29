#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  // ---------------------------------------------------------------------------
  // Helper: run LeakyReLU<N, Num, Den> on a host vector, return output vector
  // ---------------------------------------------------------------------------
  template <std::size_t N, std::int64_t Num = 1, std::int64_t Den = 100>
  static std::vector<float> run_leaky_relu(const std::vector<float>& host_input) {
    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto input_view = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    auto host_output = ffx::make_host_buffer<float[]>(N);
    auto device_input = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t tpb = 64u;
    const auto grid = ffx::make_workdiv<Acc1D>(ffx::divide_up_by(N, tpb), tpb);

    using Op = ffx::nn::LeakyReLU<N, Num, Den>;
    alpaka::exec<Acc1D>(queue, grid, Op{}, device_input.data(), device_output.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    return std::vector<float>(host_output.data(), host_output.data() + N);
  }

  // =========================================================================
  // Typed test: numerical accuracy across multiple tensor sizes
  // Default slope = 1/100 = 0.01 (matches LeakyReLU<N> template defaults)
  // =========================================================================
  template <std::size_t NumberOfElements>
  struct leaky_relu_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using leaky_relu_small_params_t = leaky_relu_params_t<16>;
  using leaky_relu_matrix_params_t = leaky_relu_params_t<8 * 32>;
  using leaky_relu_single_element_params_t = leaky_relu_params_t<1>;
  using leaky_relu_large_bulk_params_t = leaky_relu_params_t<1024>;

  template <typename T>
  class LeakyReLUTest : public ::testing::Test {};

  struct leaky_relu_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, leaky_relu_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, leaky_relu_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, leaky_relu_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, leaky_relu_large_bulk_params_t>)
        return "large_bulk";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<leaky_relu_small_params_t,
                                  leaky_relu_matrix_params_t,
                                  leaky_relu_single_element_params_t,
                                  leaky_relu_large_bulk_params_t>;
  TYPED_TEST_SUITE(LeakyReLUTest, impl_t, leaky_relu_test_name_t);

  TYPED_TEST(LeakyReLUTest, forward_random_default_slope) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    std::vector<float> host_input(N);
    std::vector<float> host_expected(N);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    constexpr float kSlope = 1.0f / 100.0f;
    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
      host_expected[i] = host_input[i] >= 0.0f ? host_input[i] : kSlope * host_input[i];
    }

    const auto output = run_leaky_relu<N>(host_input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(output[i], host_expected[i], 1e-6f) << "Mismatch at index " << i << " | input=" << host_input[i];
    }
  }

  // =========================================================================
  // Property tests — boundary / mathematical guarantees specific to LeakyReLU
  // =========================================================================

  // Positive inputs → identity (slope has no effect above 0)
  TEST(LeakyReLUPropertyTest, positive_inputs_are_identity) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(1e-4f, 100.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_leaky_relu<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], input[i], 1e-5f) << "Positive input not identity at index " << i << " input=" << input[i];
  }

  // Negative inputs → scaled by default slope 0.01
  TEST(LeakyReLUPropertyTest, negative_inputs_scaled_by_default_slope) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(11);
    std::uniform_real_distribution<float> dist(-100.0f, -1e-4f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_leaky_relu<N>(input);
    constexpr float kSlope = 0.01f;
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], kSlope * input[i], std::abs(kSlope * input[i]) * 1e-5f + 1e-7f)
          << "Negative scaling wrong at index " << i << " input=" << input[i];
  }

  // Exact zero input → exactly zero output (boundary: neither positive nor negative branch)
  TEST(LeakyReLUPropertyTest, zero_input_produces_zero_output) {
    constexpr std::size_t N = 1;
    const std::vector<float> input = {0.0f};
    const auto output = run_leaky_relu<N>(input);
    EXPECT_EQ(output[0], 0.0f);
  }

  // Non-default slope: Numerator=2, Denominator=10 → slope=0.2 (PyTorch-style)
  TEST(LeakyReLUPropertyTest, non_default_slope_0_2) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(23);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    for (auto& v : input)
      v = dist(gen);

    constexpr float kSlope = 2.0f / 10.0f;  // 0.2
    std::vector<float> expected(N);
    for (std::size_t i = 0; i < N; ++i)
      expected[i] = input[i] >= 0.0f ? input[i] : kSlope * input[i];

    // Instantiate with Numerator=2, Denominator=10
    const auto output = run_leaky_relu<N, 2, 10>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], expected[i], std::abs(expected[i]) * 1e-5f + 1e-7f)
          << "slope=0.2 mismatch at index " << i << " input=" << input[i];
  }

  // LeakyReLU output must be monotonically non-decreasing (for any fixed slope in (0,1))
  TEST(LeakyReLUPropertyTest, output_is_monotonically_non_decreasing) {
    constexpr std::size_t N = 128;
    // Strictly sorted input from -10 to +10
    std::vector<float> input(N);
    for (std::size_t i = 0; i < N; ++i)
      input[i] = -10.0f + (20.0f / (N - 1)) * static_cast<float>(i);

    const auto output = run_leaky_relu<N>(input);
    for (std::size_t i = 1; i < N; ++i)
      EXPECT_GE(output[i], output[i - 1] - 1e-6f) << "Monotonicity violated: output[" << i << "]=" << output[i]
                                                  << " < output[" << i - 1 << "]=" << output[i - 1];
  }

  // Continuity at zero: left-limit and right-limit must both converge to 0
  TEST(LeakyReLUPropertyTest, continuous_at_zero_boundary) {
    const std::vector<float> left = {-1e-5f};
    const std::vector<float> right = {+1e-5f};

    const auto out_left = run_leaky_relu<1>(left);
    const auto out_right = run_leaky_relu<1>(right);

    // Tolerance must be ≥ |x|: the positive branch is identity so f(+1e-5) = +1e-5,
    // and the negative branch gives f(-1e-5) = 0.01 * 1e-5 = 1e-7. Both are small near 0.
    EXPECT_NEAR(out_left[0], 0.0f, 2e-5f) << "Left limit at 0 not close to 0";
    EXPECT_NEAR(out_right[0], 0.0f, 2e-5f) << "Right limit at 0 not close to 0";
  }

}  // namespace ffx_runtime
