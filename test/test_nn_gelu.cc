#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  // ---------------------------------------------------------------------------
  // Helper: run GELU<N> on a host vector and return the host output vector
  // ---------------------------------------------------------------------------
  template <std::size_t N>
  static std::vector<float> run_gelu(const std::vector<float>& host_input) {
    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto input_view = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    auto host_output = ffx::make_host_buffer<float[]>(N);
    auto device_input = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t tpb = 64u;
    const auto grid = ffx::make_workdiv<Acc1D>(ffx::divide_up_by(N, tpb), tpb);
    alpaka::exec<Acc1D>(queue, grid, ffx::nn::GELU<N>{}, device_input.data(), device_output.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    return std::vector<float>(host_output.data(), host_output.data() + N);
  }

  // CPU gold: tanh-approximation GELU matching ffx::nn::functional::GELU exactly
  static float gelu_reference(const float x) {
    constexpr float sqrt_2_pi = 0.7978845608f;
    constexpr float c1 = 0.044715f;
    const float inner = sqrt_2_pi * (x + c1 * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
  }

  // =========================================================================
  // Typed test: numerical accuracy across multiple tensor sizes
  // Normal distribution mimics typical layer pre-activations
  // =========================================================================
  template <std::size_t NumberOfElements>
  struct gelu_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using gelu_small_params_t = gelu_params_t<16>;
  using gelu_matrix_params_t = gelu_params_t<8 * 32>;
  using gelu_single_element_params_t = gelu_params_t<1>;
  using gelu_large_bulk_params_t = gelu_params_t<1024>;

  template <typename T>
  class GELUTest : public ::testing::Test {};

  struct gelu_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, gelu_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, gelu_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, gelu_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, gelu_large_bulk_params_t>)
        return "large_bulk";
      return std::to_string(index);
    }
  };

  using impl_t =
      ::testing::Types<gelu_small_params_t, gelu_matrix_params_t, gelu_single_element_params_t, gelu_large_bulk_params_t>;
  TYPED_TEST_SUITE(GELUTest, impl_t, gelu_test_name_t);

  TYPED_TEST(GELUTest, forward_random) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    std::vector<float> host_input(N);
    std::vector<float> host_expected(N);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
      host_expected[i] = gelu_reference(host_input[i]);
    }

    const auto output = run_gelu<N>(host_input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(output[i], host_expected[i], 1e-4f) << "Mismatch at index " << i << " | input=" << host_input[i];
    }
  }

  // =========================================================================
  // Property tests — boundary / mathematical guarantees specific to GELU
  // =========================================================================

  // x=0 → GELU(0) = 0.5 * 0 * (1 + tanh(0)) = 0 exactly
  TEST(GELUPropertyTest, zero_input_produces_zero) {
    constexpr std::size_t N = 1;
    const auto output = run_gelu<N>({0.0f});
    EXPECT_NEAR(output[0], 0.0f, 1e-7f) << "GELU(0) must be 0";
  }

  // Large positive x: GELU(x) ≈ x (tanh → 1, so 0.5*x*(1+1) = x)
  TEST(GELUPropertyTest, large_positive_asymptotes_to_identity) {
    constexpr std::size_t N = 8;
    const std::vector<float> input(N, 10.0f);
    const auto output = run_gelu<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], 10.0f, 5e-3f) << "GELU(large positive) should asymptote to x";
  }

  // Large negative x: GELU(x) → 0 (tanh → -1, so 0.5*x*(1-1) = 0)
  TEST(GELUPropertyTest, large_negative_asymptotes_to_zero) {
    constexpr std::size_t N = 8;
    const std::vector<float> input(N, -10.0f);
    const auto output = run_gelu<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], 0.0f, 5e-3f) << "GELU(large negative) should asymptote to 0";
  }

  // GELU is non-negative for all x ≥ 0 (since 0.5*x*(1+tanh(...)) ≥ 0 when x ≥ 0)
  TEST(GELUPropertyTest, non_negative_for_positive_inputs) {
    constexpr std::size_t N = 256;
    std::vector<float> input(N);
    std::mt19937 gen(33);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_gelu<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_GE(output[i], 0.0f) << "GELU must be ≥ 0 for x=" << input[i];
  }

  // GELU output must be ≤ input for positive x (it is bounded by y=x from above)
  TEST(GELUPropertyTest, output_bounded_above_by_input_for_positive_x) {
    constexpr std::size_t N = 128;
    std::vector<float> input(N);
    std::mt19937 gen(51);
    std::uniform_real_distribution<float> dist(0.0f, 20.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_gelu<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_LE(output[i], input[i] + 1e-4f) << "GELU(x) must not exceed x for positive x=" << input[i];
  }

  // Known scalar value: GELU(1.0) ≈ 0.8413 (from PyTorch reference)
  TEST(GELUPropertyTest, known_scalar_value_at_one) {
    constexpr std::size_t N = 1;
    const auto output = run_gelu<N>({1.0f});
    EXPECT_NEAR(output[0], gelu_reference(1.0f), 1e-4f) << "GELU(1) reference mismatch";
    // Cross-check against the well-known ~0.841 value
    EXPECT_NEAR(output[0], 0.8413f, 5e-3f) << "GELU(1) should be approximately 0.8413";
  }

  // Known scalar value: GELU(-1.0) ≈ -0.1587 (from PyTorch reference)
  TEST(GELUPropertyTest, known_scalar_value_at_minus_one) {
    constexpr std::size_t N = 1;
    const auto output = run_gelu<N>({-1.0f});
    EXPECT_NEAR(output[0], gelu_reference(-1.0f), 1e-4f) << "GELU(-1) reference mismatch";
    // Must be negative (GELU dips below 0 for negative inputs)
    EXPECT_LT(output[0], 0.0f) << "GELU(-1) should be negative";
  }

}  // namespace ffx_runtime
