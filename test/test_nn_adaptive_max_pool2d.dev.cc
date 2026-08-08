#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

#include "ffx/ffx.h"

namespace ffx_runtime {

  extern "C" {
  extern const unsigned char _binary_test_nn_adaptive_max_pool2d_bin_start[];
  extern const unsigned char _binary_test_nn_adaptive_max_pool2d_bin_end[];
  }

  // ── parameter structs ─────────────────────────────────────────────────────

  template <std::size_t BatchSize, std::size_t InH, std::size_t InW, std::size_t Channels, std::size_t OutH, std::size_t OutW>
  struct adaptive_max_pool2d_params_t {
    using kernel_t = ffx::nn::AdaptiveMaxPool2d<BatchSize, InH, InW, Channels, OutH, OutW>;

    static constexpr std::size_t batch_size = BatchSize;
    static constexpr std::size_t in_height = InH;
    static constexpr std::size_t in_width = InW;
    static constexpr std::size_t channels = Channels;
    static constexpr std::size_t out_height = OutH;
    static constexpr std::size_t out_width = OutW;

    static constexpr std::size_t input_count = BatchSize * Channels * InH * InW;
    static constexpr std::size_t output_count = kernel_t::NumberOfElements;

    static constexpr std::size_t total_bytes = (input_count + output_count) * sizeof(float);
  };

  // ── test variants (must match Python TEST_CASES order exactly) ───────────

  using clean_halving_t = adaptive_max_pool2d_params_t<1, 4, 4, 1, 2, 2>;
  using multi_channel_t = adaptive_max_pool2d_params_t<1, 6, 4, 3, 2, 2>;
  using global_pool_t = adaptive_max_pool2d_params_t<2, 8, 8, 4, 1, 1>;
  using non_divisible_t = adaptive_max_pool2d_params_t<1, 7, 7, 2, 3, 3>;
  using asymmetric_output_t = adaptive_max_pool2d_params_t<1, 8, 6, 1, 4, 3>;

  using test_sequence_t =
      std::tuple<clean_halving_t, multi_channel_t, global_pool_t, non_divisible_t, asymmetric_output_t>;

  // ── blob offset resolver ─────────────────────────────────────────────────

  template <typename TCfg>
  struct AdaptiveMaxPool2dTestCase {
    template <typename TTuple, typename Target, std::size_t Index = 0>
    static constexpr std::size_t get_start_offset() {
      if constexpr (Index >= std::tuple_size_v<TTuple>)
        return 0;
      else if constexpr (std::is_same_v<std::tuple_element_t<Index, TTuple>, Target>)
        return 0;
      else
        return std::tuple_element_t<Index, TTuple>::total_bytes + get_start_offset<TTuple, Target, Index + 1>();
    }

    static constexpr std::size_t base_offset = get_start_offset<test_sequence_t, TCfg>();

    static constexpr std::size_t input_offset = base_offset;
    static constexpr std::size_t output_offset = input_offset + TCfg::input_count * sizeof(float);

    static const unsigned char* input() { return _binary_test_nn_adaptive_max_pool2d_bin_start + input_offset; }
    static const unsigned char* output() { return _binary_test_nn_adaptive_max_pool2d_bin_start + output_offset; }
  };

  // ── typed test boilerplate ────────────────────────────────────────────────

  template <typename T>
  class AdaptiveMaxPool2dTest : public ::testing::Test {};

  struct adaptive_max_pool2d_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, clean_halving_t>)
        return "clean_halving";
      if constexpr (std::is_same_v<T, multi_channel_t>)
        return "multi_channel";
      if constexpr (std::is_same_v<T, global_pool_t>)
        return "global_pool";
      if constexpr (std::is_same_v<T, non_divisible_t>)
        return "non_divisible";
      if constexpr (std::is_same_v<T, asymmetric_output_t>)
        return "asymmetric_output";
      return std::to_string(index);
    }
  };

  using impl_t =
      ::testing::Types<clean_halving_t, multi_channel_t, global_pool_t, non_divisible_t, asymmetric_output_t>;
  TYPED_TEST_SUITE(AdaptiveMaxPool2dTest, impl_t, adaptive_max_pool2d_test_name_t);

  // ── forward test ─────────────────────────────────────────────────────────

  TYPED_TEST(AdaptiveMaxPool2dTest, forward) {
    using Cfg = TypeParam;
    using Data = AdaptiveMaxPool2dTestCase<TypeParam>;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto* input_ptr = reinterpret_cast<const float*>(Data::input());
    const auto* expected_ptr = reinterpret_cast<const float*>(Data::output());

    const std::size_t input_size = Cfg::input_count;
    const std::size_t output_size = Cfg::output_count;

    const auto input_view = alpaka::createView(ffx::host(), input_ptr, static_cast<Extent>(input_size));

    auto host_output = ffx::make_host_buffer<float[]>(output_size);
    auto device_input = ffx::make_device_buffer<float[]>(queue, input_size);
    auto device_output = ffx::make_device_buffer<float[]>(queue, output_size);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t threads_per_block = 64u;
    const auto blocks = ffx::divide_up_by(output_size, threads_per_block);
    const auto grid = ffx::make_workdiv<Acc1D>(blocks, threads_per_block);

    using AdaptiveMaxPool2d = typename Cfg::kernel_t;

    alpaka::exec<Acc1D>(queue, grid, AdaptiveMaxPool2d{}, device_input.data(), device_output.data());

    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < output_size; ++i) {
      EXPECT_NEAR(host_output[i], expected_ptr[i], 1e-4f) << "Output mismatch at flat index " << i;
    }
  }

}  // namespace ffx_runtime
