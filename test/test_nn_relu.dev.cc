#include <gtest/gtest.h>
#include <random>
#include <algorithm>

#include "ffx/ffx.h"

namespace ffx_runtime {

  template <std::size_t NumberOfElements>
  struct relu_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using relu_small_params_t = relu_params_t<16>;
  using relu_matrix_params_t = relu_params_t<8 * 32>;
  using relu_single_element_params_t = relu_params_t<1>;
  using relu_large_bulk_params_t = relu_params_t<1024>;

  template <typename T>
  class ReluTest : public ::testing::Test {};

  struct relu_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, relu_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, relu_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, relu_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, relu_large_bulk_params_t>)
        return "large_bulk";
      return std::to_string(index);
    }
  };

  using impl_t =
      ::testing::Types<relu_small_params_t, relu_matrix_params_t, relu_single_element_params_t, relu_large_bulk_params_t>;
  TYPED_TEST_SUITE(ReluTest, impl_t, relu_test_name_t);

  TYPED_TEST(ReluTest, forward) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    // 1. Generate inputs and expected outputs programmatically on the host
    std::vector<float> host_input(N);
    std::vector<float> host_expected(N);

    // Use a fixed seed so tests remain perfectly reproducible
    std::mt19937 gen(42);
    // Emulate PyTorch's torch.randn(...) - 0.5 layout
    std::normal_distribution<float> dist(-0.5f, 1.0f);

    for (std::size_t i = 0; i < N; ++i) {
      host_input[i] = dist(gen);
      host_expected[i] = std::max(0.0f, host_input[i]);  // Perfect CPU gold standard
    }

    // 2. Set up Alpaka views and buffers
    const auto input_view = alpaka::createView(ffx::host(), host_input.data(), static_cast<Extent>(N));
    auto host_output = ffx::make_host_buffer<float[]>(N);

    auto device_input = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input, input_view);

    // 3. Execution Configuration
    constexpr std::size_t thread_per_block = 64u;
    const auto blocks_per_grid = ffx::divide_up_by(N, thread_per_block);
    const auto grid = ffx::make_workdiv<Acc1D>(blocks_per_grid, thread_per_block);

    using ReLUActivation = ffx::nn::ReLU<N>;

    alpaka::exec<Acc1D>(queue, grid, ReLUActivation{}, device_input.data(), device_output.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    // 4. Assert correctness
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(host_output[i], host_expected[i], 1e-6f)
          << "Mismatch noticed at index: " << i << " | Input was: " << host_input[i];
    }
  }

}  // namespace ffx_runtime