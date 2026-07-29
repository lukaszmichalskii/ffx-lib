#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

#include "ffx/ffx.h"
#include "ffx/nn/max_pool2d.h"

namespace ffx_runtime {

  extern "C" {
  extern const unsigned char _binary_test_nn_max_pool2d_bin_start[];
  extern const unsigned char _binary_test_nn_max_pool2d_bin_end[];
  }

  // ── parameter structs ─────────────────────────────────────────────────────

  template <std::size_t BatchSize,
            std::size_t InH,
            std::size_t InW,
            std::size_t Channels,
            std::size_t KH,
            std::size_t KW,
            std::size_t SH = KH,
            std::size_t SW = KW,
            std::size_t PH = 0,
            std::size_t PW = 0>
  struct max_pool2d_params_t {
    using kernel_t = ffx::nn::MaxPool2dImpl<BatchSize, InH, InW, Channels, KH, KW, SH, SW, PH, PW>;

    static constexpr std::size_t batch_size = BatchSize;
    static constexpr std::size_t in_height = InH;
    static constexpr std::size_t in_width = InW;
    static constexpr std::size_t channels = Channels;
    static constexpr std::size_t kernel_height = KH;
    static constexpr std::size_t kernel_width = KW;
    static constexpr std::size_t stride_height = SH;
    static constexpr std::size_t stride_width = SW;
    static constexpr std::size_t padding_height = PH;
    static constexpr std::size_t padding_width = PW;

    static constexpr std::size_t input_count = BatchSize * Channels * InH * InW;
    static constexpr std::size_t output_count = kernel_t::NumberOfElements;

    // byte footprint of this test case in the blob (input + output, no weights)
    static constexpr std::size_t total_bytes = (input_count + output_count) * sizeof(float);
  };

  // ── test variants (must match Python TEST_CASES order exactly) ───────────

  using basic_2x2_t = max_pool2d_params_t<1, 4, 4, 1, 2, 2>;
  using multi_channel_t = max_pool2d_params_t<1, 6, 6, 3, 2, 2>;
  using overlapping_3x3_t = max_pool2d_params_t<2, 8, 8, 4, 3, 3, 1, 1>;
  using with_padding_t = max_pool2d_params_t<1, 5, 5, 2, 2, 2, 2, 2, 1, 1>;
  using asymmetric_t = max_pool2d_params_t<1, 4, 6, 1, 2, 3>;

  using test_sequence_t = std::tuple<basic_2x2_t, multi_channel_t, overlapping_3x3_t, with_padding_t, asymmetric_t>;

  // ── blob offset resolver ─────────────────────────────────────────────────

  template <typename TCfg>
  struct MaxPool2dTestCase {
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

    static const unsigned char* input() { return _binary_test_nn_max_pool2d_bin_start + input_offset; }
    static const unsigned char* output() { return _binary_test_nn_max_pool2d_bin_start + output_offset; }
  };

  // ── typed test boilerplate ────────────────────────────────────────────────

  template <typename T>
  class MaxPool2dTest : public ::testing::Test {};

  struct max_pool2d_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, basic_2x2_t>)
        return "basic_2x2";
      if constexpr (std::is_same_v<T, multi_channel_t>)
        return "multi_channel";
      if constexpr (std::is_same_v<T, overlapping_3x3_t>)
        return "overlapping_3x3";
      if constexpr (std::is_same_v<T, with_padding_t>)
        return "with_padding";
      if constexpr (std::is_same_v<T, asymmetric_t>)
        return "asymmetric_kernel";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<basic_2x2_t, multi_channel_t, overlapping_3x3_t, with_padding_t, asymmetric_t>;
  TYPED_TEST_SUITE(MaxPool2dTest, impl_t, max_pool2d_test_name_t);

  // ── forward test ─────────────────────────────────────────────────────────

  TYPED_TEST(MaxPool2dTest, forward) {
    using Cfg = TypeParam;
    using Data = MaxPool2dTestCase<TypeParam>;

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

    using MaxPool2d = typename Cfg::kernel_t;

    alpaka::exec<Acc1D>(queue, grid, MaxPool2d{}, device_input.data(), device_output.data());

    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < output_size; ++i) {
      EXPECT_NEAR(host_output[i], expected_ptr[i], 1e-4f) << "Output mismatch at flat index " << i;
    }
  }

}  // namespace ffx_runtime
