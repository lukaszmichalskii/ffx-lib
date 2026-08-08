#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <string>

#include "ffx/ffx.h"

namespace ffx_runtime {

  template <std::size_t NumberOfElements>
  struct add_params_t {
    static constexpr std::size_t size = NumberOfElements;
  };

  using add_small_params_t = add_params_t<16>;
  using add_matrix_params_t = add_params_t<8 * 32>;
  using add_single_element_params_t = add_params_t<1>;
  using add_large_bulk_params_t = add_params_t<1024>;
  using add_non_divisible_params_t = add_params_t<1007>;

  template <typename T>
  class AddTest : public ::testing::Test {};

  struct add_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, add_small_params_t>)
        return "small";
      if constexpr (std::is_same_v<T, add_matrix_params_t>)
        return "matrix";
      if constexpr (std::is_same_v<T, add_single_element_params_t>)
        return "single_element";
      if constexpr (std::is_same_v<T, add_large_bulk_params_t>)
        return "large_bulk";
      if constexpr (std::is_same_v<T, add_non_divisible_params_t>)
        return "non_divisible";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<add_small_params_t,
                                  add_matrix_params_t,
                                  add_single_element_params_t,
                                  add_large_bulk_params_t,
                                  add_non_divisible_params_t>;

  TYPED_TEST_SUITE(AddTest, impl_t, add_test_name_t);

  TYPED_TEST(AddTest, forward) {
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
      host_expected[i] = host_input_a[i] + host_input_b[i];
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
        queue, grid, ffx::nn::Add<N>{}, device_input_a.data(), device_input_b.data(), device_output.data());

    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(host_output[i], host_expected[i], 1e-5f)
          << "Mismatch noticed at index: " << i << " | A: " << host_input_a[i] << " + B: " << host_input_b[i];
    }
  }

}  // namespace ffx_runtime