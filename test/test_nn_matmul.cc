#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

#include "ffx/ffx.h"

namespace ffx_runtime {

  extern "C" {
  extern const unsigned char _binary_test_nn_matmul_bin_start[];
  extern const unsigned char _binary_test_nn_matmul_bin_end[];
  }

  // ── parameter structs ─────────────────────────────────────────────────────

  template <std::size_t BatchSize,
            std::size_t M,
            std::size_t K,
            std::size_t N,
            std::size_t ABatchStride = M * K,
            std::size_t ARowStride = K,
            std::size_t AColStride = 1,
            std::size_t BBatchStride = K * N,
            std::size_t BRowStride = N,
            std::size_t BColStride = 1>
  struct matmul_params_t {
    using kernel_t = ffx::nn::
        MatMulImpl<BatchSize, M, K, N, ABatchStride, ARowStride, AColStride, BBatchStride, BRowStride, BColStride>;

    static constexpr std::size_t batch_size = BatchSize;
    static constexpr std::size_t m = M;
    static constexpr std::size_t k = K;
    static constexpr std::size_t n = N;

    static constexpr std::size_t a_count =
        (BatchSize - 1) * ABatchStride + (M - 1) * ARowStride + (K - 1) * AColStride + 1;
    static constexpr std::size_t b_count =
        (BatchSize - 1) * BBatchStride + (K - 1) * BRowStride + (N - 1) * BColStride + 1;
    static constexpr std::size_t c_count = BatchSize * M * N;

    static constexpr std::size_t total_bytes = (a_count + b_count + c_count) * sizeof(float);
  };

  // ── test variants ─────────────────────────────────────────────────────────

  using single_batch_t = matmul_params_t<1, 4, 8, 6>;
  using batched_standard_t = matmul_params_t<4, 16, 32, 16>;
  using asymmetric_dims_t = matmul_params_t<2, 7, 13, 5>;
  using transposed_b_t = matmul_params_t<2, 8, 16, 8, 8 * 16, 16, 1, 16 * 8, 1, 8>;

  using test_sequence_t = std::tuple<single_batch_t, batched_standard_t, asymmetric_dims_t, transposed_b_t>;

  // ── blob offset resolver ─────────────────────────────────────────────────

  template <typename TCfg>
  struct MatMulTestCase {
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

    static constexpr std::size_t a_offset = base_offset;
    static constexpr std::size_t b_offset = a_offset + TCfg::a_count * sizeof(float);
    static constexpr std::size_t c_offset = b_offset + TCfg::b_count * sizeof(float);

    static const unsigned char* a() { return _binary_test_nn_matmul_bin_start + a_offset; }
    static const unsigned char* b() { return _binary_test_nn_matmul_bin_start + b_offset; }
    static const unsigned char* c() { return _binary_test_nn_matmul_bin_start + c_offset; }
  };

  // ── typed test boilerplate ────────────────────────────────────────────────

  template <typename T>
  class MatMulTest : public ::testing::Test {};

  struct matmul_test_name_t {
    template <typename T>
    static std::string GetName(const std::size_t index) {
      if constexpr (std::is_same_v<T, single_batch_t>)
        return "single_batch";
      if constexpr (std::is_same_v<T, batched_standard_t>)
        return "batched_standard";
      if constexpr (std::is_same_v<T, asymmetric_dims_t>)
        return "asymmetric_dims";
      if constexpr (std::is_same_v<T, transposed_b_t>)
        return "transposed_b";
      return std::to_string(index);
    }
  };

  using impl_t = ::testing::Types<single_batch_t, batched_standard_t, asymmetric_dims_t, transposed_b_t>;
  TYPED_TEST_SUITE(MatMulTest, impl_t, matmul_test_name_t);

  // ── forward test ─────────────────────────────────────────────────────────

  TYPED_TEST(MatMulTest, forward) {
    using Cfg = TypeParam;
    using Data = MatMulTestCase<TypeParam>;

    const auto& device = ffx::devices<Platform>()[0];
    Queue queue{device};

    const auto* a_ptr = reinterpret_cast<const float*>(Data::a());
    const auto* b_ptr = reinterpret_cast<const float*>(Data::b());
    const auto* expected_ptr = reinterpret_cast<const float*>(Data::c());

    const auto a_view = alpaka::createView(ffx::host(), a_ptr, static_cast<Extent>(Cfg::a_count));
    const auto b_view = alpaka::createView(ffx::host(), b_ptr, static_cast<Extent>(Cfg::b_count));

    auto host_output = ffx::make_host_buffer<float[]>(Cfg::c_count);
    auto device_a = ffx::make_device_buffer<float[]>(queue, Cfg::a_count);
    auto device_b = ffx::make_device_buffer<float[]>(queue, Cfg::b_count);
    auto device_c = ffx::make_device_buffer<float[]>(queue, Cfg::c_count);

    alpaka::memcpy(queue, device_a, a_view);
    alpaka::memcpy(queue, device_b, b_view);

    constexpr std::size_t threads_per_block = 64u;
    const auto blocks = ffx::divide_up_by(Cfg::c_count, threads_per_block);
    const auto grid = ffx::make_workdiv<Acc1D>(blocks, threads_per_block);

    using MatMulKernel = typename Cfg::kernel_t;

    alpaka::exec<Acc1D>(queue, grid, MatMulKernel{}, device_a.data(), device_b.data(), device_c.data());

    alpaka::memcpy(queue, host_output, device_c);
    alpaka::wait(queue);

    for (std::size_t i = 0; i < Cfg::c_count; ++i) {
      EXPECT_NEAR(host_output[i], expected_ptr[i], 1e-4f) << "MatMul mismatch at flat index " << i;
    }
  }

}  // namespace ffx_runtime
