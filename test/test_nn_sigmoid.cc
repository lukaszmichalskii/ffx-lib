#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>

#include "ffx/ffx.h"

namespace ffx_runtime {

  // ---------------------------------------------------------------------------
  // Helper: run Sigmoid<N> on a host vector and return the host output vector
  // ---------------------------------------------------------------------------
  template <std::size_t N>
  static std::vector<float> run_sigmoid(const std::vector<float>& host_input) {
    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto input_view = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    auto host_output = ffx::make_host_buffer<float[]>(N);
    auto device_input = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t tpb = 64u;
    const auto grid = ffx::make_workdiv<Acc1D>(ffx::divide_up_by(N, tpb), tpb);
    alpaka::exec<Acc1D>(queue, grid, ffx::nn::Sigmoid<N>{}, device_input.data(), device_output.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    return std::vector<float>(host_output.data(), host_output.data() + N);
  }

  // =========================================================================
  // Typed test: numerical accuracy across multiple tensor sizes
  // Normal distribution covers deep saturation and the transition region
  // =========================================================================
  template <std::size_t NumberOfElements>
  struct sigmoid_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using sigmoid_small_params_t = sigmoid_params_t<16>;
  using sigmoid_matrix_params_t = sigmoid_params_t<8 * 32>;
  using sigmoid_single_element_params_t = sigmoid_params_t<1>;
  using sigmoid_large_bulk_params_t = sigmoid_params_t<1024>;

  template <typename T>
  class SigmoidTest : public ::testing::Test {};

  struct sigmoid_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, sigmoid_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, sigmoid_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, sigmoid_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, sigmoid_large_bulk_params_t>)
        return "large_bulk";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<sigmoid_small_params_t,
                                  sigmoid_matrix_params_t,
                                  sigmoid_single_element_params_t,
                                  sigmoid_large_bulk_params_t>;
  TYPED_TEST_SUITE(SigmoidTest, impl_t, sigmoid_test_name_t);

  TYPED_TEST(SigmoidTest, forward_random) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    std::vector<float> host_input(N);
    std::vector<float> host_expected(N);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 3.0f);

    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
      host_expected[i] = 1.0f / (1.0f + std::exp(-host_input[i]));
    }

    const auto output = run_sigmoid<N>(host_input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(output[i], host_expected[i], 1e-5f) << "Mismatch at index " << i << " | input=" << host_input[i];
    }
  }

  // =========================================================================
  // Property tests — boundary / mathematical guarantees specific to Sigmoid
  // =========================================================================

  // x=0 → exactly 0.5 (the inflection point)
  TEST(SigmoidPropertyTest, zero_input_produces_half) {
    constexpr std::size_t N = 1;
    const auto output = run_sigmoid<N>({0.0f});
    EXPECT_NEAR(output[0], 0.5f, 1e-6f) << "sigmoid(0) must be 0.5";
  }

  // Output is always in [0, 1].
  // Note: float32 saturates to exactly 0.0f or 1.0f for extreme inputs (|x| > ~88),
  // so the mathematical open interval (0, 1) becomes a closed [0, 1] in finite precision.
  TEST(SigmoidPropertyTest, output_always_in_unit_interval) {
    constexpr std::size_t N = 512;
    std::vector<float> input(N);
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    for (auto& v : input)
      v = dist(gen);

    const auto output = run_sigmoid<N>(input);
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_GE(output[i], 0.0f) << "Output below 0 at index " << i;
      EXPECT_LE(output[i], 1.0f) << "Output above 1 at index " << i;
    }
  }

  // Symmetry property: σ(-x) = 1 - σ(x) for all x
  // This catches sign-flip bugs in the exponential argument
  TEST(SigmoidPropertyTest, symmetry_sigma_neg_x_equals_one_minus_sigma_x) {
    constexpr std::size_t N = 128;
    std::vector<float> pos_input(N), neg_input(N);
    std::mt19937 gen(55);
    std::uniform_real_distribution<float> dist(1e-3f, 10.0f);
    for (std::size_t i = 0; i < N; ++i) {
      pos_input[i] = dist(gen);
      neg_input[i] = -pos_input[i];
    }

    const auto pos_out = run_sigmoid<N>(pos_input);
    const auto neg_out = run_sigmoid<N>(neg_input);

    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(neg_out[i], 1.0f - pos_out[i], 1e-5f)
          << "Symmetry σ(-x)=1-σ(x) violated at index " << i << " x=" << pos_input[i];
  }

  // Saturation: very large positive x → output close to 1
  TEST(SigmoidPropertyTest, large_positive_saturates_to_one) {
    constexpr std::size_t N = 8;
    const std::vector<float> input(N, 100.0f);
    const auto output = run_sigmoid<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], 1.0f, 1e-4f) << "Expected saturation to 1 at index " << i;
  }

  // Saturation: very large negative x → output close to 0
  TEST(SigmoidPropertyTest, large_negative_saturates_to_zero) {
    constexpr std::size_t N = 8;
    const std::vector<float> input(N, -100.0f);
    const auto output = run_sigmoid<N>(input);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(output[i], 0.0f, 1e-4f) << "Expected saturation to 0 at index " << i;
  }

  // Monotonically non-decreasing — sigmoid is a strictly increasing function
  TEST(SigmoidPropertyTest, output_is_monotonically_non_decreasing) {
    constexpr std::size_t N = 128;
    std::vector<float> input(N);
    for (std::size_t i = 0; i < N; ++i)
      input[i] = -10.0f + (20.0f / (N - 1)) * static_cast<float>(i);

    const auto output = run_sigmoid<N>(input);
    for (std::size_t i = 1; i < N; ++i)
      EXPECT_GE(output[i], output[i - 1] - 1e-7f) << "Monotonicity violated at index " << i;
  }

}  // namespace ffx_runtime
