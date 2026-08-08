#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  // ---------------------------------------------------------------------------
  // Helper: run Hardswish<N> on a host vector and return the host output vector
  // ---------------------------------------------------------------------------
  template <std::size_t N>
  static std::vector<float> run_hardswish(const std::vector<float>& host_input) {
    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto input_view = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    auto host_output = ffx::make_host_buffer<float[]>(N);
    auto device_input = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t tpb = 64u;
    const auto grid = ffx::make_workdiv<Acc1D>(ffx::divide_up_by(N, tpb), tpb);
    alpaka::exec<Acc1D>(queue, grid, ffx::nn::Hardswish<N>{}, device_input.data(), device_output.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    return std::vector<float>(host_output.data(), host_output.data() + N);
  }

  // CPU gold: hardswish(x) = x * relu6(x+3) / 6 — three piecewise regions:
  //   x ≤ -3  →  0
  //   -3 < x < 3  →  x * (x + 3) / 6
  //   x ≥ 3   →  x
  static float hardswish_reference(const float x) {
    const float relu6_val = std::min(6.0f, std::max(0.0f, x + 3.0f));
    return x * relu6_val / 6.0f;
  }

  // =========================================================================
  // Typed test: numerical accuracy across multiple tensor sizes
  // Uniform [-6, 6] covers all three piecewise regions
  // =========================================================================
  template <std::size_t NumberOfElements>
  struct hardswish_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using hardswish_small_params_t = hardswish_params_t<16>;
  using hardswish_matrix_params_t = hardswish_params_t<8 * 32>;
  using hardswish_single_element_params_t = hardswish_params_t<1>;
  using hardswish_large_bulk_params_t = hardswish_params_t<1024>;

  template <typename T>
  class HardswishTest : public ::testing::Test {};

  struct hardswish_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, hardswish_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, hardswish_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, hardswish_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, hardswish_large_bulk_params_t>)
        return "large_bulk";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<hardswish_small_params_t,
                                  hardswish_matrix_params_t,
                                  hardswish_single_element_params_t,
                                  hardswish_large_bulk_params_t>;
  TYPED_TEST_SUITE(HardswishTest, impl_t, hardswish_test_name_t);

  TYPED_TEST(HardswishTest, forward_random) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    std::vector<float> host_input(N);
    std::vector<float> host_expected(N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);

    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
      host_expected[i] = hardswish_reference(host_input[i]);
    }

    const auto output = run_hardswish<N>(host_input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(output[i], host_expected[i], 1e-6f) << "Mismatch at index " << i << " | input=" << host_input[i];
    }
  }

  // =========================================================================
  // Property tests — piecewise region guarantees specific to Hardswish
  // =========================================================================

  // Region 1: x ≤ -3 → output is identically 0
  TEST(HardswishPropertyTest, below_minus_three_produces_zero) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-100.0f, -3.001f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_hardswish<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(output[i], 0.0f) << "Expected 0 for x=" << input[i] << " (x ≤ -3 region)";
  }

  // Region 3: x ≥ 3 → output equals input (identity)
  TEST(HardswishPropertyTest, above_three_is_identity) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(11);
    std::uniform_real_distribution<float> dist(3.001f, 100.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_hardswish<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], input[i], std::abs(input[i]) * 1e-6f + 1e-6f)
          << "Expected identity for x=" << input[i] << " (x ≥ 3 region)";
  }

  // Exact boundary values: x=-3 → 0, x=3 → 3
  TEST(HardswishPropertyTest, exact_boundary_values) {
    constexpr std::size_t N = 2;
    const std::vector<float> input = {-3.0f, 3.0f};
    const auto output = run_hardswish<N>(input);
    EXPECT_NEAR(output[0], 0.0f, 1e-6f) << "hardswish(-3) must be 0";
    EXPECT_NEAR(output[1], 3.0f, 1e-5f) << "hardswish(3) must be 3";
  }

  // x=0 → hardswish(0) = 0 * relu6(3)/6 = 0 * 1/2 = 0
  TEST(HardswishPropertyTest, zero_input_produces_zero) {
    constexpr std::size_t N = 1;
    const auto output = run_hardswish<N>({0.0f});
    EXPECT_NEAR(output[0], 0.0f, 1e-7f) << "hardswish(0) must be 0";
  }

  // Output is non-negative for all x ≥ 0 (x ≥ 0 and relu6(x+3)/6 > 0)
  TEST(HardswishPropertyTest, non_negative_for_non_negative_inputs) {
    constexpr std::size_t N = 256;
    std::vector<float> input(N);
    std::mt19937 gen(33);
    std::uniform_real_distribution<float> dist(0.0f, 50.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_hardswish<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_GE(output[i], 0.0f) << "Hardswish must be ≥ 0 for x=" << input[i];
  }

  // Piecewise quadratic region: verify formula x*(x+3)/6 for x in (-3, 3)
  TEST(HardswishPropertyTest, quadratic_region_formula) {
    constexpr std::size_t N = 64;
    std::vector<float> input(N);
    std::mt19937 gen(19);
    std::uniform_real_distribution<float> dist(-2.999f, 2.999f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_hardswish<N>(input);
    for (std::size_t i = 0; i < N; ++i) {
      const float expected = input[i] * (input[i] + 3.0f) / 6.0f;
      EXPECT_NEAR(output[i], expected, 1e-5f) << "Quadratic region formula x*(x+3)/6 violated at x=" << input[i];
    }
  }

  // Continuity at x=-3: left and right limits converge to 0
  TEST(HardswishPropertyTest, continuous_at_minus_three) {
    constexpr std::size_t N = 1;
    const auto out_left = run_hardswish<N>({-3.001f});
    const auto out_right = run_hardswish<N>({-2.999f});
    EXPECT_NEAR(out_left[0], 0.0f, 1e-3f) << "Left of -3 should be near 0";
    EXPECT_NEAR(out_right[0], 0.0f, 1e-3f) << "Right of -3 should be near 0";
  }

  // Continuity at x=3: left and right limits converge to 3
  TEST(HardswishPropertyTest, continuous_at_three) {
    constexpr std::size_t N = 1;
    const auto out_left = run_hardswish<N>({2.999f});
    const auto out_right = run_hardswish<N>({3.001f});
    EXPECT_NEAR(out_left[0], 3.0f, 1e-2f) << "Left of 3 should be near 3";
    EXPECT_NEAR(out_right[0], 3.0f, 1e-2f) << "Right of 3 should be near 3";
  }

}  // namespace ffx_runtime
