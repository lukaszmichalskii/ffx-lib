#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>
#include <string>

#include "ffx/ffx.h"

namespace ffx_runtime {

  extern "C" {
  extern const unsigned char _binary_test_nn_conv2d_bin_start[];
  extern const unsigned char _binary_test_nn_conv2d_bin_end[];
  }

  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  struct conv2d_params_t {
    using conv2d_t = ffx::nn::Conv2d<BatchSize,
                                     InHeight,
                                     InWidth,
                                     InChannels,
                                     OutChannels,
                                     KernelHeight,
                                     KernelWidth,
                                     StrideHeight,
                                     StrideWidth,
                                     PaddingHeight,
                                     PaddingWidth>;

    static constexpr std::size_t weight_count = OutChannels * InChannels * KernelHeight * KernelWidth;
    static constexpr std::size_t bias_count = OutChannels;
    static constexpr std::size_t input_count = BatchSize * InChannels * InHeight * InWidth;
    static constexpr std::size_t output_count = conv2d_t::NumberOfElements;

    static constexpr std::size_t total_bytes = (weight_count + bias_count + input_count + output_count) * sizeof(float);
  };

  using conv2d_asymmetric_params_t = conv2d_params_t<1, 5, 3, 2, 3, 3, 2, 2, 1, 1, 0>;
  using conv2d_multi_batch_params_t = conv2d_params_t<3, 8, 8, 8, 16, 3, 3, 1, 1, 1, 1>;
  using conv2d_pointwise_params_t = conv2d_params_t<1, 14, 14, 4, 8, 1, 1, 1, 1, 0, 0>;
  using conv2d_valid_stride_params_t = conv2d_params_t<1, 7, 7, 1, 1, 3, 3, 3, 3, 0, 0>;

  // Ordered layout matching how Python streams out byte data blocks
  using test_sequence_t = std::tuple<conv2d_asymmetric_params_t,
                                     conv2d_multi_batch_params_t,
                                     conv2d_pointwise_params_t,
                                     conv2d_valid_stride_params_t>;

  template <typename TCfg>
  struct Conv2dTestCase {
    // Structural accumulation of previous elements block steps
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

    // Layer sub-offsets mapping onto memory
    static constexpr std::size_t weight_offset = base_offset;
    static constexpr std::size_t bias_offset = weight_offset + (TCfg::weight_count * sizeof(float));
    static constexpr std::size_t input_offset = bias_offset + (TCfg::bias_count * sizeof(float));
    static constexpr std::size_t output_offset = input_offset + (TCfg::input_count * sizeof(float));

    static const unsigned char* weight() { return _binary_test_nn_conv2d_bin_start + weight_offset; }
    static const unsigned char* bias() { return _binary_test_nn_conv2d_bin_start + bias_offset; }
    static const unsigned char* input() { return _binary_test_nn_conv2d_bin_start + input_offset; }
    static const unsigned char* output() { return _binary_test_nn_conv2d_bin_start + output_offset; }
  };

  template <typename T>
  class Conv2dTest : public ::testing::Test {};

  struct conv2d_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, conv2d_asymmetric_params_t>)
        return "asymmetric";
      if constexpr (std::is_same_v<T, conv2d_multi_batch_params_t>)
        return "multi_batch";
      if constexpr (std::is_same_v<T, conv2d_pointwise_params_t>)
        return "pointwise";
      if constexpr (std::is_same_v<T, conv2d_valid_stride_params_t>)
        return "valid_stride";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<conv2d_asymmetric_params_t,
                                  conv2d_multi_batch_params_t,
                                  conv2d_pointwise_params_t,
                                  conv2d_valid_stride_params_t>;
  TYPED_TEST_SUITE(Conv2dTest, impl_t, conv2d_test_name_t);

  TYPED_TEST(Conv2dTest, forward) {
    using Cfg = TypeParam;
    using Data = Conv2dTestCase<TypeParam>;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto* weights_ptr = reinterpret_cast<const float*>(Data::weight());
    const auto* bias_ptr = reinterpret_cast<const float*>(Data::bias());
    const auto* input_ptr = reinterpret_cast<const float*>(Data::input());
    const auto* expected_ptr = reinterpret_cast<const float*>(Data::output());

    const std::size_t weights_size = Cfg::weight_count;
    const std::size_t bias_size = Cfg::bias_count;
    const std::size_t input_size = Cfg::input_count;
    const std::size_t output_size = Cfg::output_count;

    const auto weights_view = alpaka::createView(ffx::host(), weights_ptr, static_cast<Extent>(weights_size));
    const auto bias_view = alpaka::createView(ffx::host(), bias_ptr, static_cast<Extent>(bias_size));
    const auto input_view = alpaka::createView(ffx::host(), input_ptr, static_cast<Extent>(input_size));

    auto host_output = ffx::make_host_buffer<float[]>(output_size);

    auto device_input = ffx::make_device_buffer<float[]>(queue, input_size);
    auto device_weights = ffx::make_device_buffer<float[]>(queue, weights_size);
    auto device_bias = ffx::make_device_buffer<float[]>(queue, bias_size);
    auto device_output = ffx::make_device_buffer<float[]>(queue, output_size);

    alpaka::memcpy(queue, device_input, input_view);
    alpaka::memcpy(queue, device_weights, weights_view);
    alpaka::memcpy(queue, device_bias, bias_view);

    constexpr std::size_t thread_per_block = 64u;
    const auto blocks_per_grid = ffx::divide_up_by(output_size, thread_per_block);
    const auto grid = ffx::make_workdiv<Acc1D>(blocks_per_grid, thread_per_block);

    using Conv2d = Cfg::conv2d_t;

    alpaka::exec<Acc1D>(
        queue, grid, Conv2d{}, device_input.data(), device_output.data(), device_weights.data(), device_bias.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < output_size; ++i) {
      EXPECT_NEAR(host_output[i], expected_ptr[i], 1e-3f)
          << "Failure occurs within Type Variant Offset Flat Position Matrix Index Context: " << i;
    }
  }

}  // namespace ffx_runtime