#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "test_soa_definition_custom_methods.h"

// Helper HIP error checking macro for gTest
#define HIP_CHECK(condition)                                                   \
  do {                                                                         \
    hipError_t error = condition;                                              \
    ASSERT_EQ(error, hipSuccess) << "HIP error: " << hipGetErrorString(error); \
  } while (0)

// HIP Kernels
__global__ void calculateNorm(SoAConstView soaConstView, float* resultNorm, double* resultVelNorm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= static_cast<int>(soaConstView.metadata().size()))
    return;

  resultNorm[i] = soaConstView[i].square_norm_position();
  resultVelNorm[i] = soaConstView[i].square_norm_velocity();
}

__global__ void checkNormalise(SoAView soaView, double* checkTimesFunction) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= static_cast<int>(soaView.metadata().size()))
    return;

  checkTimesFunction[i] = SoAView::const_element::time(soaView[i].x(), soaView[i].v_x());
  soaView[i].normalise();
}

// Test Fixture for HIP Customized Methods Tests
class SoACustomizedMethodsHipTest : public ::testing::Test {
protected:
  static constexpr std::size_t elems = 10;

  void SetUp() override {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No HIP/ROCm devices available. Skipping HIP tests.";
    }

    bufferSize = SoA::computeDataSize(elems);

    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_buf), bufferSize));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_buf), bufferSize));

    h_soahdLayout = std::make_unique<SoA>(h_buf, elems);
    h_view = std::make_unique<SoAView>(*h_soahdLayout);
    h_Constview = std::make_unique<SoAConstView>(*h_soahdLayout);

    d_soahdLayout = std::make_unique<SoA>(d_buf, elems);
    d_view = std::make_unique<SoAView>(*d_soahdLayout);
    d_Constview = std::make_unique<SoAConstView>(*d_soahdLayout);

    // Fill up host data
    for (std::size_t i = 0; i < elems; ++i) {
      (*h_view)[i].x() = static_cast<float>(i);
      (*h_view)[i].y() = static_cast<float>(i) * 2.0f;
      (*h_view)[i].z() = static_cast<float>(i) * 3.0f;
      (*h_view)[i].v_x() = static_cast<double>(i);
      (*h_view)[i].v_y() = static_cast<double>(i) * 20.0;
      (*h_view)[i].v_z() = static_cast<double>(i) * 30.0;
    }
    h_view->detectorType() = 42;

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_position_norms), elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_velocity_norms), elems * sizeof(double)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_times), elems * sizeof(double)));

    // Host -> Device copy
    HIP_CHECK(hipMemcpy(d_buf, h_buf, bufferSize, hipMemcpyHostToDevice));
  }

  void TearDown() override {
    if (d_position_norms)
      (void)hipFree(d_position_norms);
    if (d_velocity_norms)
      (void)hipFree(d_velocity_norms);
    if (d_times)
      (void)hipFree(d_times);
    if (d_buf)
      (void)hipFree(d_buf);
    if (h_buf)
      (void)hipHostFree(h_buf);
  }

  std::size_t bufferSize{0};
  std::byte* h_buf{nullptr};
  std::byte* d_buf{nullptr};

  float* d_position_norms{nullptr};
  double* d_velocity_norms{nullptr};
  double* d_times{nullptr};

  std::unique_ptr<SoA> h_soahdLayout;
  std::unique_ptr<SoAView> h_view;
  std::unique_ptr<SoAConstView> h_Constview;

  std::unique_ptr<SoA> d_soahdLayout;
  std::unique_ptr<SoAView> d_view;
  std::unique_ptr<SoAConstView> d_Constview;
};

TEST_F(SoACustomizedMethodsHipTest, ConstViewMethodsHip) {
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (elems + threadsPerBlock - 1) / threadsPerBlock;

  hipLaunchKernelGGL(
      calculateNorm, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, *d_Constview, d_position_norms, d_velocity_norms);

  std::vector<float> h_position_norms(elems);
  std::vector<double> h_velocity_norms(elems);

  HIP_CHECK(hipMemcpy(h_position_norms.data(), d_position_norms, elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_velocity_norms.data(), d_velocity_norms, elems * sizeof(double), hipMemcpyDeviceToHost));

  // Check correctness of square_norm() functions
  for (std::size_t i = 0; i < elems; ++i) {
    const float position_norm =
        std::sqrt((*h_Constview)[i].x() * (*h_Constview)[i].x() + (*h_Constview)[i].y() * (*h_Constview)[i].y() +
                  (*h_Constview)[i].z() * (*h_Constview)[i].z());
    const double velocity_norm = std::sqrt((*h_Constview)[i].v_x() * (*h_Constview)[i].v_x() +
                                           (*h_Constview)[i].v_y() * (*h_Constview)[i].v_y() +
                                           (*h_Constview)[i].v_z() * (*h_Constview)[i].v_z());

    EXPECT_FLOAT_EQ(h_position_norms[i], position_norm);
    EXPECT_DOUBLE_EQ(h_velocity_norms[i], velocity_norm);
  }
}

TEST_F(SoACustomizedMethodsHipTest, ViewMethodsHip) {
  std::array<double, elems> times;
  times[0] = 0.0;
  for (std::size_t i = 1; i < elems; ++i) {
    times[i] = (*h_view)[i].x() / (*h_view)[i].v_x();
  }

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (elems + threadsPerBlock - 1) / threadsPerBlock;

  hipLaunchKernelGGL(checkNormalise, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, *d_view, d_times);

  std::vector<double> h_times(elems);
  HIP_CHECK(hipMemcpy(h_times.data(), d_times, elems * sizeof(double), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_buf, d_buf, bufferSize, hipMemcpyDeviceToHost));

  // Check correctness of time() function
  for (std::size_t i = 0; i < elems; ++i) {
    EXPECT_DOUBLE_EQ(h_times[i], times[i]);
  }

  EXPECT_FLOAT_EQ((*h_view)[0].square_norm_position(), 0.0f);
  EXPECT_DOUBLE_EQ((*h_view)[0].square_norm_velocity(), 0.0);

  for (std::size_t i = 1; i < elems; ++i) {
    EXPECT_NEAR((*h_view)[i].square_norm_position(), 1.0f, 1e-6f);
    EXPECT_NEAR((*h_view)[i].square_norm_velocity(), 1.0, 1e-9);
  }
}