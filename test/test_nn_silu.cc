#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  // ---------------------------------------------------------------------------
  // Helper: run SiLU<N> on a host vector and return the host output vector
  // ---------------------------------------------------------------------------
  template <std::size_t N>
  static std::vector<float> run_silu(const std::vector<float>& host_input) {
    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto input_view = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    auto host_output = ffx::make_host_buffer<float[]>(N);
    auto device_input = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t tpb = 64u;
    const auto grid = ffx::make_workdiv<Acc1D>(ffx::divide_up_by(N, tpb), tpb);
    alpaka::exec<Acc1D>(queue, grid, ffx::nn::SiLU<N>{}, device_input.data(), device_output.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    return std::vector<float>(host_output.data(), host_output.data() + N);
  }

  // SiLU reference: x * sigmoid(x) = x / (1 + exp(-x))
  static float silu_reference(const float x) { return x / (1.0f + std::exp(-x)); }

  // =========================================================================
  // Typed test: numerical accuracy across multiple tensor sizes
  // Normal distribution covers the full positive/negative landscape
  // =========================================================================
  template <std::size_t NumberOfElements>
  struct silu_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using silu_small_params_t = silu_params_t<16>;
  using silu_matrix_params_t = silu_params_t<8 * 32>;
  using silu_single_element_params_t = silu_params_t<1>;
  using silu_large_bulk_params_t = silu_params_t<1024>;

  template <typename T>
  class SiLUTest : public ::testing::Test {};

  struct silu_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, silu_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, silu_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, silu_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, silu_large_bulk_params_t>)
        return "large_bulk";
      return std::to_string(index);
    }
  };

  using impl_t =
      ::testing::Types<silu_small_params_t, silu_matrix_params_t, silu_single_element_params_t, silu_large_bulk_params_t>;
  TYPED_TEST_SUITE(SiLUTest, impl_t, silu_test_name_t);

  TYPED_TEST(SiLUTest, forward_random) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    std::vector<float> host_input(N);
    std::vector<float> host_expected(N);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
      host_expected[i] = silu_reference(host_input[i]);
    }

    const auto output = run_silu<N>(host_input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(output[i], host_expected[i], 1e-5f) << "Mismatch at index " << i << " | input=" << host_input[i];
    }
  }

  // =========================================================================
  // Property tests — boundary / mathematical guarantees specific to SiLU
  // =========================================================================

  // x=0 → silu(0) = 0 * 0.5 = 0 exactly
  TEST(SiLUPropertyTest, zero_input_produces_zero) {
    constexpr std::size_t N = 1;
    const auto output = run_silu<N>({0.0f});
    EXPECT_NEAR(output[0], 0.0f, 1e-7f) << "silu(0) must be 0";
  }

  // For large positive x: SiLU(x) ≈ x (since sigmoid(x) → 1)
  TEST(SiLUPropertyTest, large_positive_asymptotes_to_identity) {
    constexpr std::size_t N = 8;
    const std::vector<float> input(N, 50.0f);
    const auto output = run_silu<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], 50.0f, 1e-3f) << "SiLU should asymptote to identity for large positive x";
  }

  // For large negative x: SiLU(x) → 0 (since x * sigmoid(x) → x * 0 → 0)
  TEST(SiLUPropertyTest, large_negative_asymptotes_to_zero) {
    constexpr std::size_t N = 8;
    const std::vector<float> input(N, -50.0f);
    const auto output = run_silu<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], 0.0f, 1e-3f) << "SiLU should asymptote to 0 for large negative x";
  }

  // SiLU has a global minimum for a small negative x (≈ -1.278)
  // At x ≈ -1.278, SiLU(x) ≈ -0.2785
  // This test ensures the implementation correctly handles the trough region
  TEST(SiLUPropertyTest, known_minimum_trough) {
    constexpr std::size_t N = 1;
    // SiLU(-1.278) ≈ -0.2785
    const auto output = run_silu<N>({-1.278f});
    EXPECT_NEAR(output[0], silu_reference(-1.278f), 1e-4f) << "Known trough value mismatch at x=-1.278";
    // And it must be negative (the only region where SiLU dips below 0)
    EXPECT_LT(output[0], 0.0f) << "SiLU output at x=-1.278 should be negative";
  }

  // SiLU is non-negative for all x ≥ 0 (since x ≥ 0 and sigmoid(x) > 0)
  TEST(SiLUPropertyTest, non_negative_for_positive_inputs) {
    constexpr std::size_t N = 256;
    std::vector<float> input(N);
    std::mt19937 gen(33);
    std::uniform_real_distribution<float> dist(0.0f, 20.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_silu<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_GE(output[i], 0.0f) << "SiLU must be ≥ 0 for x=" << input[i];
  }

  // SiLU(x) = x * sigmoid(x) — verify the decomposed identity holds
  TEST(SiLUPropertyTest, equals_x_times_sigmoid) {
    constexpr std::size_t N = 128;
    std::vector<float> input(N);
    std::mt19937 gen(77);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_silu<N>(input);
    for (std::size_t i = 0; i < N; ++i) {
      const float expected = input[i] * (1.0f / (1.0f + std::exp(-input[i])));
      EXPECT_NEAR(output[i], expected, 1e-5f) << "SiLU(x) ≠ x*sigmoid(x) at index " << i << " x=" << input[i];
    }
  }

}  // namespace ffx_runtime
