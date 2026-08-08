#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <string>

#include "ffx/ffx.h"

namespace ffx_runtime {

  template <std::size_t NumberOfElements>
  struct mul_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using mul_small_params_t = mul_params_t<16>;
  using mul_matrix_params_t = mul_params_t<8 * 32>;
  using mul_single_element_params_t = mul_params_t<1>;
  using mul_large_bulk_params_t = mul_params_t<1024>;
  using mul_non_divisible_params_t = mul_params_t<1007>;

  template <typename T>
  class MulTest : public ::testing::Test {};

  struct mul_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, mul_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, mul_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, mul_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, mul_large_bulk_params_t>)
        return "large_bulk";
      if constexpr (std::is_same_v<T, mul_non_divisible_params_t>)
        return "non_divisible";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<mul_small_params_t,
                                  mul_matrix_params_t,
                                  mul_single_element_params_t,
                                  mul_large_bulk_params_t,
                                  mul_non_divisible_params_t>;

  TYPED_TEST_SUITE(MulTest, impl_t, mul_test_name_t);

  TYPED_TEST(MulTest, forward) {
    using Cfg = TypeParam;
    constexpr std::size_t N = Cfg::size;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    std::vector<float> host_input_a(N);
    std::vector<float> host_input_b(N);
    std::vector<float> host_expected(N);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < N; ++i) {
      host_input_a[i] = dist(gen);
      host_input_b[i] = dist(gen);
      host_expected[i] = host_input_a[i] * host_input_b[i];
    }

    const auto view_a = alpaka::createView(ffx::host(), host_input_a.data(), static_cast<Extent>(N));
    const auto view_b = alpaka::createView(ffx::host(), host_input_b.data(), static_cast<Extent>(N));

    auto host_output = ffx::make_host_buffer<float[]>(N);

    auto device_input_a = ffx::make_device_buffer<float[]>(queue, N);
    auto device_input_b = ffx::make_device_buffer<float[]>(queue, N);
    auto device_output = ffx::make_device_buffer<float[]>(queue, N);

    alpaka::memcpy(queue, device_input_a, view_a);
    alpaka::memcpy(queue, device_input_b, view_b);

    constexpr std::size_t thread_per_block = 64u;
    const auto blocks_per_grid = ffx::divide_up_by(N, thread_per_block);
    const auto grid = ffx::make_workdiv<Acc1D>(blocks_per_grid, thread_per_block);

    alpaka::exec<Acc1D>(
        queue, grid, ffx::nn::Mul<N>{}, device_input_a.data(), device_input_b.data(), device_output.data());

    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(host_output[i], host_expected[i], 1e-5f)
          << "Mismatch noticed at index: " << i << " | A: " << host_input_a[i] << " * B: " << host_input_b[i];
    }
  }

}  // namespace ffx_runtime