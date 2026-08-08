#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "ffx/ffx.h"

// Helper HIP error checking macro for gTest
#define HIP_CHECK(condition)                                                   \
  do {                                                                         \
    hipError_t error = condition;                                              \
    ASSERT_EQ(error, hipSuccess) << "HIP error: " << hipGetErrorString(error); \
  } while (0)

// SoA Layout Declarations
GENERATE_SOA_LAYOUT(SoAHostDeviceLayoutTemplate,
                    SOA_COLUMN(double, x),
                    SOA_COLUMN(double, y),
                    SOA_COLUMN(double, z),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, r),
                    SOA_SCALAR(const char*, description),
                    SOA_SCALAR(uint32_t, someNumber))

using SoAHostDeviceLayout = SoAHostDeviceLayoutTemplate<>;
using SoAHostDeviceView = SoAHostDeviceLayout::View;
using SoAHostDeviceRangeCheckingView =
    SoAHostDeviceLayout::ViewTemplate<ffx::soa::restrict_qualify::enabled, ffx::soa::range_checking::enabled>;
using SoAHostDeviceConstView = SoAHostDeviceLayout::ConstView;

GENERATE_SOA_LAYOUT(SoADeviceOnlyLayoutTemplate,
                    SOA_COLUMN(uint16_t, color),
                    SOA_COLUMN(double, value),
                    SOA_COLUMN(double*, py),
                    SOA_COLUMN(uint32_t, count),
                    SOA_COLUMN(uint32_t, anotherCount))

using SoADeviceOnlyLayout = SoADeviceOnlyLayoutTemplate<>;
using SoADeviceOnlyView = SoADeviceOnlyLayout::View;

GENERATE_SOA_LAYOUT(SoAFullDeviceLayoutTemplate,
                    SOA_COLUMN(double, x),
                    SOA_COLUMN(double, y),
                    SOA_COLUMN(double, z),
                    SOA_COLUMN(uint16_t, color),
                    SOA_COLUMN(double, value),
                    SOA_COLUMN(double*, py),
                    SOA_COLUMN(uint32_t, count),
                    SOA_COLUMN(uint32_t, anotherCount),
                    SOA_SCALAR(const char*, description),
                    SOA_SCALAR(uint32_t, someNumber))

using SoAFullDeviceLayout =
    SoAFullDeviceLayoutTemplate<ffx::soa::CacheLineSize::Gpu, ffx::soa::AlignmentEnforcement::enforced>;
using SoAFullDeviceView = SoAFullDeviceLayout::View;
using SoAFullDeviceConstView = SoAFullDeviceLayout::ConstView;

// Validation layout for zero-column / scalar-only case
GENERATE_SOA_LAYOUT(TestSoALayoutNoColumn, SOA_SCALAR(double, r))
GENERATE_SOA_LAYOUT(TestSoALayoutNoColumn2, SOA_SCALAR(double, r), SOA_SCALAR(double, r2))

// HIP Kernels
__global__ void crossProductKernel(SoAHostDeviceView soa, const int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numElements)
    return;
  auto si = soa[i];
  si.r() = si.a().cross(si.b());
}

__global__ void producerKernel(SoAFullDeviceView soa, const int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numElements)
    return;
  auto si = soa[i];
  si.color() &= 0x55 << (i % (sizeof(si.color()) - sizeof(char)));
  si.value() = sqrt(si.x() * si.x() + si.y() * si.y() + si.z() * si.z());
}

__global__ void consumerKernel(SoAFullDeviceView soa, const int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numElements)
    return;
  auto si = soa[i];
  si.x() = si.color() * si.value();
}

using RangeCheckingHostDeviceView =
    SoAHostDeviceLayout::ViewTemplate<SoAHostDeviceView::restrictQualify, ffx::soa::range_checking::enabled>;

__global__ void rangeCheckKernel(RangeCheckingHostDeviceView soa) {
  [[maybe_unused]] auto si = soa[soa.metadata().size()];
}

// gTest Fixture for HIP Integration Tests
class SoAHipTest : public ::testing::Test {
protected:
  static constexpr unsigned int numElements = 65537;  // Non-aligned count

  void SetUp() override {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No HIP/ROCm devices available. Skipping HIP tests.";
    }

    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    hostDeviceSize = SoAHostDeviceLayout::computeDataSize(numElements);
    deviceOnlySize = SoADeviceOnlyLayout::computeDataSize(numElements);

    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_buf), hostDeviceSize));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&d_buf), hostDeviceSize + deviceOnlySize));
  }

  void TearDown() override {
    if (h_buf)
      HIP_CHECK(hipHostFree(h_buf));
    if (d_buf)
      HIP_CHECK(hipHostFree(d_buf));
    if (stream)
      HIP_CHECK(hipStreamDestroy(stream));
  }

  hipStream_t stream{nullptr};
  std::size_t hostDeviceSize{0};
  std::size_t deviceOnlySize{0};
  std::byte* h_buf{nullptr};
  std::byte* d_buf{nullptr};
};

TEST_F(SoAHipTest, ColumnAlignments) {
  SoAHostDeviceLayout h_layout(h_buf, numElements);
  SoAHostDeviceView h_view(h_layout);

  SoAHostDeviceLayout d_layout(d_buf, numElements);
  SoADeviceOnlyLayout d_only_layout(d_layout.metadata().nextByte(), numElements);
  SoAHostDeviceView d_hd_view(d_layout);
  SoADeviceOnlyView d_do_view(d_only_layout);

  const auto d_hd_recs = d_hd_view.records();
  const auto d_do_recs = d_do_view.records();

  SoAFullDeviceView d_full_view(d_hd_recs.x(),
                                d_hd_recs.y(),
                                d_hd_recs.z(),
                                d_do_recs.color(),
                                d_do_recs.value(),
                                d_do_recs.py(),
                                d_do_recs.count(),
                                d_do_recs.anotherCount(),
                                d_hd_recs.description(),
                                d_hd_recs.someNumber());

  // Host alignment checks
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(h_view.metadata().addressOf_x()) % SoAHostDeviceView::alignment);
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(h_view.metadata().addressOf_y()) % SoAHostDeviceView::alignment);
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(h_view.metadata().addressOf_z()) % SoAHostDeviceView::alignment);
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(h_view.metadata().addressOf_a()) % SoAHostDeviceView::alignment);
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(h_view.metadata().addressOf_b()) % SoAHostDeviceView::alignment);
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(h_view.metadata().addressOf_r()) % SoAHostDeviceView::alignment);

  // Composite device view alignment checks
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(d_full_view.metadata().addressOf_x()) % SoAFullDeviceView::alignment);
  EXPECT_EQ(0u,
            reinterpret_cast<std::uintptr_t>(d_full_view.metadata().addressOf_color()) % SoAFullDeviceView::alignment);
  EXPECT_EQ(0u,
            reinterpret_cast<std::uintptr_t>(d_full_view.metadata().addressOf_value()) % SoAFullDeviceView::alignment);
}

TEST_F(SoAHipTest, EndToEndHostDevicePipeline) {
  SoAHostDeviceLayout h_layout(h_buf, numElements);
  SoAHostDeviceView h_view(h_layout);

  SoAHostDeviceLayout d_hd_layout(d_buf, numElements);
  SoADeviceOnlyLayout d_do_layout(d_hd_layout.metadata().nextByte(), numElements);
  SoAHostDeviceView d_hd_view(d_hd_layout);
  SoADeviceOnlyView d_do_view(d_do_layout);

  const auto d_hd_recs = d_hd_view.records();
  const auto d_do_recs = d_do_view.records();

  SoAFullDeviceView d_full_view(d_hd_recs.x(),
                                d_hd_recs.y(),
                                d_hd_recs.z(),
                                d_do_recs.color(),
                                d_do_recs.value(),
                                d_do_recs.py(),
                                d_do_recs.count(),
                                d_do_recs.anotherCount(),
                                d_hd_recs.description(),
                                d_hd_recs.someNumber());

  // Fill host buffer
  std::memset(h_layout.metadata().data(), 0, hostDeviceSize);
  for (std::size_t i = 0; i < numElements; ++i) {
    auto si = h_view[i];
    double v1 = 1.0 * i + 1.0;
    double v2 = 2.0 * i;
    double v3 = 3.0 * i - 1.0;
    if (i % 2) {
      si = {v1, v2, v3, {v1, v2, v3}, {v3, v2, v1}, {0, 0, 0}};
    } else {
      si.x() = si.a()(0) = si.b()(2) = v1;
      si.y() = si.a()(1) = si.b()(1) = v2;
      si.z() = si.a()(2) = si.b()(0) = v3;
    }
  }
  h_view.someNumber() = numElements + 2;

  // Copy to device
  HIP_CHECK(hipMemcpyAsync(d_buf, h_buf, hostDeviceSize, hipMemcpyDefault, stream));

  // Run kernels
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

  hipLaunchKernelGGL(crossProductKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, d_hd_view, numElements);
  HIP_CHECK(hipMemsetAsync(d_do_layout.metadata().data(), 0xFF, d_do_layout.metadata().byteSize(), stream));
  hipLaunchKernelGGL(producerKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, d_full_view, numElements);
  hipLaunchKernelGGL(consumerKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, d_full_view, numElements);

  // Fetch back to host
  HIP_CHECK(hipMemcpyAsync(h_buf, d_buf, hostDeviceSize, hipMemcpyDefault, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify computations
  SoAHostDeviceConstView h_const_view(h_layout);
  for (std::size_t i = 0; i < numElements; ++i) {
    auto si = h_const_view[i];

    // Check Eigen cross product calculation
    Eigen::Vector3d expectedCross = si.a().cross(si.b());
    EXPECT_DOUBLE_EQ(si.r()(0), expectedCross(0));
    EXPECT_DOUBLE_EQ(si.r()(1), expectedCross(1));
    EXPECT_DOUBLE_EQ(si.r()(2), expectedCross(2));

    // Check pipeline calculation
    double initialX = 1.0 * i + 1.0;
    double initialY = 2.0 * i;
    double initialZ = 3.0 * i - 1.0;
    std::uint16_t expectedColor = 0x55 << (i % (sizeof(std::uint16_t) - sizeof(char)));
    double expectedX = expectedColor * std::sqrt(initialX * initialX + initialY * initialY + initialZ * initialZ);

    double relDiff = std::abs(si.x() - expectedX) / expectedX;
    EXPECT_LT(relDiff, 2.0 * std::numeric_limits<double>::epsilon()) << "Pipeline calculation mismatch at index " << i;
  }
}

TEST_F(SoAHipTest, HostRangeCheckingExceptions) {
  SoAHostDeviceLayout h_layout(h_buf, numElements);
  SoAHostDeviceRangeCheckingView rc_view(h_layout);

  // Check bounds exceptions on Host
  EXPECT_THROW([[maybe_unused]] auto si = rc_view[rc_view.metadata().size()], std::out_of_range);
  EXPECT_THROW([[maybe_unused]] auto si = rc_view[-1], std::out_of_range);

  // Valid accesses should pass
  EXPECT_NO_THROW([[maybe_unused]] auto si = rc_view[rc_view.metadata().size() - 1]);
  EXPECT_NO_THROW([[maybe_unused]] auto si = rc_view[0]);
}

// Disabled until ROCm provides a non-fatal way to assert in device code
TEST_F(SoAHipTest, DeviceKernelRangeCheckingFailure) {
  GTEST_SKIP() << "ROCm/HIP device range checking kernel trap is currently non-fatal/unsupported.";
  SoAHostDeviceLayout d_hd_layout(d_buf, numElements);
  RangeCheckingHostDeviceView rc_device_view(d_hd_layout);

  hipLaunchKernelGGL(rangeCheckKernel, dim3(1), dim3(1), 0, stream, rc_device_view);

  hipError_t err = hipStreamSynchronize(stream);
  EXPECT_NE(err, hipSuccess) << "Device kernel out-of-bounds access should have caused a HIP error failure.";

  HIP_CHECK(hipGetLastError());
}