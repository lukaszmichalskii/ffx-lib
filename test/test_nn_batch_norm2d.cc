#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>
#include <string>

#include "ffx/ffx.h"

namespace ffx_runtime {

  extern "C" {
  extern const unsigned char _binary_test_nn_batch_norm2d_bin_start[];
  extern const unsigned char _binary_test_nn_batch_norm2d_bin_end[];
  }

  template <std::size_t BatchSize, std::size_t Channels, std::size_t Height, std::size_t Width>
  struct bn2d_params_t {
    static constexpr std::size_t batch_size = BatchSize;
    static constexpr std::size_t channels = Channels;
    static constexpr std::size_t height = Height;
    static constexpr std::size_t width = Width;

    // Per-channel parameter arrays (gamma, beta, mean, variance)
    static constexpr std::size_t param_count = Channels;
    // Spatial feature map (NCHW layout)
    static constexpr std::size_t feature_count = BatchSize * Channels * Height * Width;

    // Binary layout: gamma[C] + beta[C] + mean[C] + var[C] + input[N,C,H,W] + output[N,C,H,W]
    static constexpr std::size_t total_bytes = (4 * param_count + 2 * feature_count) * sizeof(float);
  };

  // Test configurations — must exactly match TEST_CASES in test_batch_norm2d.py
  using bn2d_standard_params_t = bn2d_params_t<1, 4, 4, 4>;
  using bn2d_multi_batch_params_t = bn2d_params_t<2, 8, 4, 4>;
  using bn2d_single_pixel_params_t = bn2d_params_t<1, 16, 1, 1>;
  using bn2d_deep_params_t = bn2d_params_t<1, 32, 3, 3>;

  // Ordered sequence matching the python write order
  using test_sequence_t =
      std::tuple<bn2d_standard_params_t, bn2d_multi_batch_params_t, bn2d_single_pixel_params_t, bn2d_deep_params_t>;

  template <typename TCfg>
  struct BatchNorm2dTestCase {
    template <typename TTuple, typename Target, std::size_t Index = 0>
    static constexpr std::size_t get_start_offset() {
      if constexpr (Index >= std::tuple_size_v<TTuple>) {
        return 0;
      } else if constexpr (std::is_same_v<std::tuple_element_t<Index, TTuple>, Target>) {
        return 0;
      } else {
        return std::tuple_element_t<Index, TTuple>::total_bytes + get_start_offset<TTuple, Target, Index + 1>();
      }
    }

    static constexpr std::size_t base_offset = get_start_offset<test_sequence_t, TCfg>();

    // Sub-offsets within one test block — order matches Python write order:
    //   gamma, beta, running_mean, running_var, input, output
    static constexpr std::size_t gamma_offset = base_offset;
    static constexpr std::size_t beta_offset = gamma_offset + TCfg::param_count * sizeof(float);
    static constexpr std::size_t mean_offset = beta_offset + TCfg::param_count * sizeof(float);
    static constexpr std::size_t var_offset = mean_offset + TCfg::param_count * sizeof(float);
    static constexpr std::size_t input_offset = var_offset + TCfg::param_count * sizeof(float);
    static constexpr std::size_t output_offset = input_offset + TCfg::feature_count * sizeof(float);

    static const unsigned char* gamma() { return _binary_test_nn_batch_norm2d_bin_start + gamma_offset; }
    static const unsigned char* beta() { return _binary_test_nn_batch_norm2d_bin_start + beta_offset; }
    static const unsigned char* mean() { return _binary_test_nn_batch_norm2d_bin_start + mean_offset; }
    static const unsigned char* var() { return _binary_test_nn_batch_norm2d_bin_start + var_offset; }
    static const unsigned char* input() { return _binary_test_nn_batch_norm2d_bin_start + input_offset; }
    static const unsigned char* output() { return _binary_test_nn_batch_norm2d_bin_start + output_offset; }
  };

  template <typename T>
  class BatchNorm2dTest : public ::testing::Test {};

  struct bn2d_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, bn2d_standard_params_t>)
        return "standard";
      if constexpr (std::is_same_v<T, bn2d_multi_batch_params_t>)
        return "multi_batch";
      if constexpr (std::is_same_v<T, bn2d_single_pixel_params_t>)
        return "single_pixel";
      if constexpr (std::is_same_v<T, bn2d_deep_params_t>)
        return "deep";
      return std::to_string(index);
    }
  };

  using impl_t =
      ::testing::Types<bn2d_standard_params_t, bn2d_multi_batch_params_t, bn2d_single_pixel_params_t, bn2d_deep_params_t>;
  TYPED_TEST_SUITE(BatchNorm2dTest, impl_t, bn2d_test_name_t);

  TYPED_TEST(BatchNorm2dTest, forward) {
    using Cfg = TypeParam;
    using Data = BatchNorm2dTestCase<TypeParam>;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto* gamma_ptr = reinterpret_cast<const float*>(Data::gamma());
    const auto* beta_ptr = reinterpret_cast<const float*>(Data::beta());
    const auto* mean_ptr = reinterpret_cast<const float*>(Data::mean());
    const auto* var_ptr = reinterpret_cast<const float*>(Data::var());
    const auto* input_ptr = reinterpret_cast<const float*>(Data::input());
    const auto* expected_ptr = reinterpret_cast<const float*>(Data::output());

    const std::size_t param_size = Cfg::param_count;
    const std::size_t feature_size = Cfg::feature_count;

    const auto gamma_view = alpaka::createView(ffx::host(), gamma_ptr, static_cast<Extent>(param_size));
    const auto beta_view = alpaka::createView(ffx::host(), beta_ptr, static_cast<Extent>(param_size));
    const auto mean_view = alpaka::createView(ffx::host(), mean_ptr, static_cast<Extent>(param_size));
    const auto var_view = alpaka::createView(ffx::host(), var_ptr, static_cast<Extent>(param_size));
    const auto input_view = alpaka::createView(ffx::host(), input_ptr, static_cast<Extent>(feature_size));

    auto host_output = ffx::make_host_buffer<float[]>(feature_size);

    auto device_gamma = ffx::make_device_buffer<float[]>(queue, param_size);
    auto device_beta = ffx::make_device_buffer<float[]>(queue, param_size);
    auto device_mean = ffx::make_device_buffer<float[]>(queue, param_size);
    auto device_var = ffx::make_device_buffer<float[]>(queue, param_size);
    auto device_input = ffx::make_device_buffer<float[]>(queue, feature_size);
    auto device_output = ffx::make_device_buffer<float[]>(queue, feature_size);

    alpaka::memcpy(queue, device_gamma, gamma_view);
    alpaka::memcpy(queue, device_beta, beta_view);
    alpaka::memcpy(queue, device_mean, mean_view);
    alpaka::memcpy(queue, device_var, var_view);
    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t thread_per_block = 64u;
    const auto blocks_per_grid = ffx::divide_up_by(feature_size, thread_per_block);
    const auto grid = ffx::make_workdiv<Acc1D>(blocks_per_grid, thread_per_block);

    // Default epsilon: 1/100000 = 1e-5, matching PyTorch's default BatchNorm2d eps
    using BN2d = ffx::nn::BatchNorm2d<Cfg::batch_size, Cfg::channels, Cfg::height, Cfg::width>;

    alpaka::exec<Acc1D>(queue,
                        grid,
                        BN2d{},
                        device_input.data(),
                        device_output.data(),
                        device_gamma.data(),
                        device_beta.data(),
                        device_mean.data(),
                        device_var.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < feature_size; ++i) {
      EXPECT_NEAR(host_output[i], expected_ptr[i], 1e-4f) << "Mismatch at flat index: " << i;
    }
  }

}  // namespace ffx_runtime
