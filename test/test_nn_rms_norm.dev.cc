#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

#include "ffx/ffx.h"

namespace ffx_runtime {

  extern "C" {
  extern const unsigned char _binary_test_nn_rms_norm_bin_start[];
  extern const unsigned char _binary_test_nn_rms_norm_bin_end[];
  }

  // ── parameter structs ─────────────────────────────────────────────────────

  template <std::size_t Tokens,
            std::size_t EmbedDim,
            bool HasGamma = true,
            std::int64_t EpsilonNumerator = 1,
            std::int64_t EpsilonDenominator = 1000000>
  struct rms_norm_params_t {
    using kernel_t = ffx::nn::RMSNormImpl<Tokens, EmbedDim, EpsilonNumerator, EpsilonDenominator>;

    static constexpr std::size_t num_tokens = Tokens;
    static constexpr std::size_t embedding_dim = EmbedDim;
    static constexpr bool has_gamma = HasGamma;

    static constexpr std::size_t input_count = Tokens * EmbedDim;
    static constexpr std::size_t gamma_count = HasGamma ? EmbedDim : 0;
    static constexpr std::size_t output_count = input_count;

    static constexpr std::size_t total_bytes = (input_count + gamma_count + output_count) * sizeof(float);
  };

  // ── test variants (must match Python TEST_CASES order exactly) ───────────

  using standard_affine_t = rms_norm_params_t<4, 64, true>;
  using single_token_t = rms_norm_params_t<1, 128, true>;
  using no_affine_t = rms_norm_params_t<8, 32, false>;
  using large_embedding_t = rms_norm_params_t<2, 512, true>;
  using custom_eps_t = rms_norm_params_t<16, 16, true, 1, 100000>;

  using test_sequence_t = std::tuple<standard_affine_t, single_token_t, no_affine_t, large_embedding_t, custom_eps_t>;

  // ── blob offset resolver ─────────────────────────────────────────────────

  template <typename TCfg>
  struct RMSNormTestCase {
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
    static constexpr std::size_t gamma_offset = input_offset + TCfg::input_count * sizeof(float);
    static constexpr std::size_t output_offset = gamma_offset + TCfg::gamma_count * sizeof(float);

    static const unsigned char* input() { return _binary_test_nn_rms_norm_bin_start + input_offset; }
    static const unsigned char* gamma() {
      return TCfg::has_gamma ? (_binary_test_nn_rms_norm_bin_start + gamma_offset) : nullptr;
    }
    static const unsigned char* output() { return _binary_test_nn_rms_norm_bin_start + output_offset; }
  };

  // ── typed test boilerplate ────────────────────────────────────────────────

  template <typename T>
  class RMSNormTest : public ::testing::Test {};

  struct rms_norm_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, standard_affine_t>)
        return "standard_affine";
      if constexpr (std::is_same_v<T, single_token_t>)
        return "single_token";
      if constexpr (std::is_same_v<T, no_affine_t>)
        return "no_affine";
      if constexpr (std::is_same_v<T, large_embedding_t>)
        return "large_embedding";
      if constexpr (std::is_same_v<T, custom_eps_t>)
        return "custom_eps";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<standard_affine_t, single_token_t, no_affine_t, large_embedding_t, custom_eps_t>;
  TYPED_TEST_SUITE(RMSNormTest, impl_t, rms_norm_test_name_t);

  // ── forward test ─────────────────────────────────────────────────────────

  TYPED_TEST(RMSNormTest, forward) {
    using Cfg = TypeParam;
    using Data = RMSNormTestCase<TypeParam>;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto* input_ptr = reinterpret_cast<const float*>(Data::input());
    const auto* gamma_ptr = reinterpret_cast<const float*>(Data::gamma());
    const auto* expected_ptr = reinterpret_cast<const float*>(Data::output());

    const std::size_t input_size = Cfg::input_count;
    const std::size_t gamma_size = Cfg::gamma_count;
    const std::size_t output_size = Cfg::output_count;

    const auto input_view = alpaka::createView(ffx::host(), input_ptr, static_cast<Extent>(input_size));

    auto host_output = ffx::make_host_buffer<float[]>(output_size);
    auto device_input = ffx::make_device_buffer<float[]>(queue, input_size);
    auto device_output = ffx::make_device_buffer<float[]>(queue, output_size);

    alpaka::memcpy(queue, device_input, input_view);

    float* d_gamma_ptr = nullptr;
    auto device_gamma = ffx::make_device_buffer<float[]>(queue, Cfg::has_gamma ? gamma_size : 1);
    if constexpr (Cfg::has_gamma) {
      const auto gamma_view = alpaka::createView(ffx::host(), gamma_ptr, static_cast<Extent>(gamma_size));
      alpaka::memcpy(queue, device_gamma, gamma_view);
      d_gamma_ptr = device_gamma.data();
    }

    constexpr std::size_t threads_per_block = 64u;
    const auto blocks = ffx::divide_up_by(Cfg::num_tokens, threads_per_block);
    const auto grid = ffx::make_workdiv<Acc1D>(blocks, threads_per_block);

    using RMSNorm = typename Cfg::kernel_t;

    alpaka::exec<Acc1D>(queue, grid, RMSNorm{}, device_input.data(), device_output.data(), d_gamma_ptr);

    alpaka::memcpy(queue, host_output, device_output);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < output_size; ++i) {
      EXPECT_NEAR(host_output[i], expected_ptr[i], 1e-4f) << "Output mismatch at flat index " << i;
    }
  }

}  // namespace ffx_runtime
