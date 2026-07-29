#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  // ---------------------------------------------------------------------------
  // Helper: run Tanh<N> on a host vector and return the host output vector
  // ---------------------------------------------------------------------------
  template <std::size_t N>
  static std::vector<float> run_tanh(const std::vector<float>& host_input) {
    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto input_view = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    auto host_output = ffx::make_host_buffer<float[]>(N);
    auto device_input = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t tpb = 64u;
    const auto grid = ffx::make_workdiv<Acc1D>(ffx::divide_up_by(N, tpb), tpb);
    alpaka::exec<Acc1D>(queue, grid, ffx::nn::Tanh<N>{}, device_input.data(), device_output.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    return std::vector<float>(host_output.data(), host_output.data() + N);
  }

  // =========================================================================
  // Typed test: numerical accuracy across multiple tensor sizes
  // Uniform [-4, 4] exercises saturation at both extremes and the linear center
  // =========================================================================
  template <std::size_t NumberOfElements>
  struct tanh_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using tanh_small_params_t = tanh_params_t<16>;
  using tanh_matrix_params_t = tanh_params_t<8 * 32>;
  using tanh_single_element_params_t = tanh_params_t<1>;
  using tanh_large_bulk_params_t = tanh_params_t<1024>;

  template <typename T>
  class TanhTest : public ::testing::Test {};

  struct tanh_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, tanh_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, tanh_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, tanh_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, tanh_large_bulk_params_t>)
        return "large_bulk";
      return std::to_string(index);
    }
  };

  using impl_t =
      ::testing::Types<tanh_small_params_t, tanh_matrix_params_t, tanh_single_element_params_t, tanh_large_bulk_params_t>;
  TYPED_TEST_SUITE(TanhTest, impl_t, tanh_test_name_t);

  TYPED_TEST(TanhTest, forward_random) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    std::vector<float> host_input(N);
    std::vector<float> host_expected(N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
      host_expected[i] = std::tanh(host_input[i]);
    }

    const auto output = run_tanh<N>(host_input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(output[i], host_expected[i], 1e-5f) << "Mismatch at index " << i << " | input=" << host_input[i];
    }
  }

  // =========================================================================
  // Property tests — boundary / mathematical guarantees specific to Tanh
  // =========================================================================

  // x=0 → exactly 0 (tanh is an odd function with tanh(0)=0)
  TEST(TanhPropertyTest, zero_input_produces_zero) {
    constexpr std::size_t N = 1;
    const auto output = run_tanh<N>({0.0f});
    EXPECT_NEAR(output[0], 0.0f, 1e-7f) << "tanh(0) must be 0";
  }

  // Output is always in [-1, 1].
  // Note: float32 saturates to exactly ±1.0f for extreme inputs (|x| > ~18),
  // so the mathematical open interval (-1, 1) becomes a closed [-1, 1] in finite precision.
  TEST(TanhPropertyTest, output_always_in_closed_minus_one_to_one) {
    constexpr std::size_t N = 512;
    std::vector<float> input(N);
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_tanh<N>(input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_GE(output[i], -1.0f) << "Output below -1 at index " << i;
      EXPECT_LE(output[i], 1.0f) << "Output above  1 at index " << i;
    }
  }

  // Anti-symmetry (odd function): tanh(-x) = -tanh(x) for all x
  // Catches sign errors in the exponential implementation
  TEST(TanhPropertyTest, antisymmetry_tanh_neg_x_equals_neg_tanh_x) {
    constexpr std::size_t N = 128;
    std::vector<float> pos_input(N), neg_input(N);
    std::mt19937 gen(55);
    std::uniform_real_distribution<float> dist(1e-3f, 5.0f);
    for (std::size_t i = 0; i < N; ++i) {
      pos_input[i] = dist(gen);
      neg_input[i] = -pos_input[i];
    }

    const auto pos_out = run_tanh<N>(pos_input);
    const auto neg_out = run_tanh<N>(neg_input);

    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(neg_out[i], -pos_out[i], 1e-5f)
          << "Anti-symmetry tanh(-x)=-tanh(x) violated at index " << i << " x=" << pos_input[i];
  }

  // Positive saturation: very large positive x → output close to +1
  TEST(TanhPropertyTest, large_positive_saturates_to_plus_one) {
    constexpr std::size_t N = 8;
    const std::vector<float> input(N, 50.0f);
    const auto output = run_tanh<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], 1.0f, 1e-5f) << "Expected +1 saturation at index " << i;
  }

  // Negative saturation: very large negative x → output close to -1
  TEST(TanhPropertyTest, large_negative_saturates_to_minus_one) {
    constexpr std::size_t N = 8;
    const std::vector<float> input(N, -50.0f);
    const auto output = run_tanh<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], -1.0f, 1e-5f) << "Expected -1 saturation at index " << i;
  }

  // Monotonically non-decreasing — tanh is strictly increasing
  TEST(TanhPropertyTest, output_is_monotonically_non_decreasing) {
    constexpr std::size_t N = 128;
    std::vector<float> input(N);
    for (std::size_t i = 0; i < N; ++i)
      input[i] = -10.0f + (20.0f / (N - 1)) * static_cast<float>(i);

    const auto output = run_tanh<N>(input);
    for (std::size_t i = 1; i < N; ++i)
      EXPECT_GE(output[i], output[i - 1] - 1e-7f) << "Monotonicity violated at index " << i;
  }

  // Near-linear region: for small |x|, tanh(x) ≈ x (slope ≈ 1 at origin)
  TEST(TanhPropertyTest, near_linear_for_small_inputs) {
    constexpr std::size_t N = 16;
    std::vector<float> input(N);
    for (std::size_t i = 0; i < N; ++i)
      input[i] = -0.1f + (0.2f / (N - 1)) * static_cast<float>(i);

    const auto output = run_tanh<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      // tanh(x) ≈ x with error < x^3/3 for |x| < 0.1 → error < 0.0003
      EXPECT_NEAR(output[i], input[i], 5e-4f) << "Near-linear approximation failed at input=" << input[i];
  }

}  // namespace ffx_runtime
