#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

#include "ffx/ffx.h"

namespace ffx_runtime {

  extern "C" {
  extern const unsigned char _binary_test_nn_softmax_bin_start[];
  extern const unsigned char _binary_test_nn_softmax_bin_end[];
  }

  // ── parameter structs ─────────────────────────────────────────────────────

  template <std::size_t NumReductions,
            std::size_t ReductionSize,
            std::size_t StrideWithinDim = 1,
            std::size_t OuterStride = ReductionSize * StrideWithinDim,
            typename TAlias = ffx::nn::SoftmaxImpl<NumReductions, ReductionSize, StrideWithinDim, OuterStride>>
  struct softmax_params_t {
    using kernel_t = TAlias;

    static constexpr std::size_t num_reductions = NumReductions;
    static constexpr std::size_t reduction_size = ReductionSize;
    static constexpr std::size_t stride_within_dim = StrideWithinDim;
    static constexpr std::size_t outer_stride = OuterStride;

    static constexpr std::size_t total_elements = NumReductions * OuterStride;
    static constexpr std::size_t total_bytes = (total_elements + total_elements) * sizeof(float);
  };

  // ── test variants matching aliases and Python order ──────────────────────

  using softmax_1d_t = softmax_params_t<1, 100, 1, 100, ffx::nn::Softmax1D<100>>;
  using softmax_2d_t = softmax_params_t<8, 64, 1, 64, ffx::nn::Softmax2D<8, 64>>;
  using softmax_3d_t = softmax_params_t<2 * 16, 128, 1, 128, ffx::nn::Softmax3D<2, 16, 128>>;
  using softmax_4d_t = softmax_params_t<2 * 8 * 32, 32, 1, 32, ffx::nn::Softmax4D<2, 8, 32, 32>>;
  using strided_custom_t = softmax_params_t<4, 16, 2, 64>;

  using test_sequence_t = std::tuple<softmax_1d_t, softmax_2d_t, softmax_3d_t, softmax_4d_t, strided_custom_t>;

  // ── blob offset resolver ─────────────────────────────────────────────────

  template <typename TCfg>
  struct SoftmaxTestCase {
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
    static constexpr std::size_t output_offset = input_offset + TCfg::total_elements * sizeof(float);

    static const unsigned char* input() { return _binary_test_nn_softmax_bin_start + input_offset; }
    static const unsigned char* output() { return _binary_test_nn_softmax_bin_start + output_offset; }
  };

  // ── typed test boilerplate ────────────────────────────────────────────────

  template <typename T>
  class SoftmaxTest : public ::testing::Test {};

  struct softmax_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, softmax_1d_t>)
        return "softmax_1d";
      if constexpr (std::is_same_v<T, softmax_2d_t>)
        return "softmax_2d";
      if constexpr (std::is_same_v<T, softmax_3d_t>)
        return "softmax_3d";
      if constexpr (std::is_same_v<T, softmax_4d_t>)
        return "softmax_4d";
      if constexpr (std::is_same_v<T, strided_custom_t>)
        return "strided_custom";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<softmax_1d_t, softmax_2d_t, softmax_3d_t, softmax_4d_t, strided_custom_t>;
  TYPED_TEST_SUITE(SoftmaxTest, impl_t, softmax_test_name_t);

  // ── forward test ─────────────────────────────────────────────────────────

  TYPED_TEST(SoftmaxTest, forward) {
    using Cfg = TypeParam;
    using Data = SoftmaxTestCase<TypeParam>;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto* input_ptr = reinterpret_cast<const float*>(Data::input());
    const auto* expected_ptr = reinterpret_cast<const float*>(Data::output());

    const std::size_t total_elements = Cfg::total_elements;

    const auto input_view = alpaka::createView(ffx::host(), input_ptr, static_cast<Extent>(total_elements));

    auto host_output = ffx::make_host_buffer<float[]>(total_elements);
    auto device_input = ffx::make_device_buffer<float[]>(queue, total_elements);
    auto device_output = ffx::make_device_buffer<float[]>(queue, total_elements);

    alpaka::memcpy(queue, device_input, input_view);

    constexpr std::size_t threads_per_block = 64u;
    const auto blocks = ffx::divide_up_by(Cfg::num_reductions, threads_per_block);
    const auto grid = ffx::make_workdiv<Acc1D>(blocks, threads_per_block);

    using SoftmaxKernel = typename Cfg::kernel_t;

    alpaka::exec<Acc1D>(queue, grid, SoftmaxKernel{}, device_input.data(), device_output.data());

    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t r = 0; r < Cfg::num_reductions; ++r) {
      const auto base = r * Cfg::outer_stride;
      for (std::size_t i = 0; i < Cfg::reduction_size; ++i) {
        const auto idx = base + i * Cfg::stride_within_dim;
        EXPECT_NEAR(host_output[idx], expected_ptr[idx], 1e-4f)
            << "Softmax mismatch at reduction vector " << r << ", element " << i << " (flat index " << idx << ")";
      }
    }
  }

}  // namespace ffx_runtime