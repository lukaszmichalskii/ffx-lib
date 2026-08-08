#include <gtest/gtest.h>
#include "ffx/ffx.h"

namespace ffx_runtime {

  extern "C" {
  extern const unsigned char _binary_test_nn_linear_bin_start[];
  extern const unsigned char _binary_test_nn_linear_bin_end[];
  }

  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  struct linear_params_t {
    static constexpr std::size_t batch_size = BatchSize;
    static constexpr std::size_t in_dim = InDim;
    static constexpr std::size_t out_dim = OutDim;

    static constexpr std::size_t weight_count = OutDim * InDim;
    static constexpr std::size_t bias_count = OutDim;
    static constexpr std::size_t input_count = BatchSize * InDim;
    static constexpr std::size_t output_count = BatchSize * OutDim;

    static constexpr std::size_t total_bytes = (weight_count + bias_count + input_count + output_count) * sizeof(float);
  };

  using linear_standard_params_t = linear_params_t<1, 128, 64>;
  using linear_wide_multi_batch_params_t = linear_params_t<4, 32, 512>;
  using linear_single_unit_params_t = linear_params_t<1, 1, 1>;
  using linear_3d_tensor_params_t = linear_params_t<10, 64, 32>;

  template <typename... TConfigs>
  struct offset_calculator;

  template <>
  struct offset_calculator<> {
    static constexpr std::size_t value = 0;
  };

  template <typename THead, typename... TTail>
  struct offset_calculator<THead, TTail...> {
    static constexpr std::size_t value = THead::total_bytes + offset_calculator<TTail...>::value;
  };

  // ordered sequence of test cases in the binary payload
  using test_sequence_t = std::tuple<linear_standard_params_t,
                                     linear_wide_multi_batch_params_t,
                                     linear_single_unit_params_t,
                                     linear_3d_tensor_params_t>;

  template <typename TCfg>
  struct LinearTestCase {
    // find where test case block begins within the tuple sequence
    template <typename TTuple, typename Target, std::size_t Index = 0>
    static constexpr std::size_t get_start_offset() {
      if constexpr (Index >= std::tuple_size_v<TTuple>) {
        return 0;
      } else if constexpr (std::is_same_v<std::tuple_element_t<Index, TTuple>, Target>) {
        return 0;  // target block, do not accumulate size
      } else {
        return std::tuple_element_t<Index, TTuple>::total_bytes + get_start_offset<TTuple, Target, Index + 1>();
      }
    }

    static constexpr std::size_t base_offset = get_start_offset<test_sequence_t, TCfg>();

    // relative sub-offsets have to match the exact writing layout from python script
    static constexpr std::size_t weight_offset = base_offset;
    static constexpr std::size_t bias_offset = weight_offset + TCfg::weight_count * sizeof(float);
    static constexpr std::size_t input_offset = bias_offset + TCfg::bias_count * sizeof(float);
    static constexpr std::size_t output_offset = input_offset + TCfg::input_count * sizeof(float);

    // pointer accessors mapping onto the compiled unified byte array
    static const unsigned char* weight() { return _binary_test_nn_linear_bin_start + weight_offset; }
    static const unsigned char* bias() { return _binary_test_nn_linear_bin_start + bias_offset; }
    static const unsigned char* input() { return _binary_test_nn_linear_bin_start + input_offset; }
    static const unsigned char* output() { return _binary_test_nn_linear_bin_start + output_offset; }
  };

  template <typename T>
  class LinearTest : public ::testing::Test {};

  struct linear_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, linear_standard_params_t>)
        return "standard";
      if constexpr (std::is_same_v<T, linear_wide_multi_batch_params_t>)
        return "wide_multi_batch";
      if constexpr (std::is_same_v<T, linear_single_unit_params_t>)
        return "single_unit";
      if constexpr (std::is_same_v<T, linear_3d_tensor_params_t>)
        return "3d_tensor";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<linear_standard_params_t,
                                  linear_wide_multi_batch_params_t,
                                  linear_single_unit_params_t,
                                  linear_3d_tensor_params_t>;
  TYPED_TEST_SUITE(LinearTest, impl_t, linear_test_name_t);

  TYPED_TEST(LinearTest, forward) {
    using Cfg = TypeParam;
    using Data = LinearTestCase<TypeParam>;

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

    using Linear = ffx::nn::Linear<Cfg::batch_size, Cfg::in_dim, Cfg::out_dim>;

    alpaka::exec<Acc1D>(
        queue, grid, Linear{}, device_input.data(), device_output.data(), device_weights.data(), device_bias.data());
    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < output_size; ++i) {
      EXPECT_NEAR(host_output[i], expected_ptr[i], 1e-3f)
          << "Failure within Linear Variant Flat Array Index Location: " << i;
    }
  }

}  // namespace ffx_runtime
