#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "test_soa_definition_custom_methods.h"

// Helper CUDA error checking macro for gTest
#define CUDA_CHECK(condition)                                               \
  do {                                                                      \
    cudaError_t error = condition;                                          \
    ASSERT_EQ(error, cudaSuccess)                                           \
        << "CUDA error: " << cudaGetErrorString(error);                     \
  } while (0)

// CUDA Kernels
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

// Test Fixture for CUDA Customized Methods Tests
class SoACustomizedMethodsCudaTest : public ::testing::Test {
protected:
  static constexpr std::size_t elems = 10;

  void SetUp() override {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices available. Skipping CUDA tests.";
    }

    bufferSize = SoA::computeDataSize(elems);

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_buf), bufferSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buf), bufferSize));

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

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_position_norms), elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_velocity_norms), elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_times), elems * sizeof(double)));

    // Host -> Device copy
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bufferSize, cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    if (d_position_norms) (void)cudaFree(d_position_norms);
    if (d_velocity_norms) (void)cudaFree(d_velocity_norms);
    if (d_times) (void)cudaFree(d_times);
    if (d_buf) (void)cudaFree(d_buf);
    if (h_buf) (void)cudaFreeHost(h_buf);
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

TEST_F(SoACustomizedMethodsCudaTest, ConstViewMethodsCuda) {
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (elems + threadsPerBlock - 1) / threadsPerBlock;

  calculateNorm<<<blocksPerGrid, threadsPerBlock>>>(*d_Constview, d_position_norms, d_velocity_norms);
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> h_position_norms(elems);
  std::vector<double> h_velocity_norms(elems);

  CUDA_CHECK(cudaMemcpy(h_position_norms.data(), d_position_norms, elems * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_velocity_norms.data(), d_velocity_norms, elems * sizeof(double), cudaMemcpyDeviceToHost));

  // Check correctness of square_norm() functions
  for (std::size_t i = 0; i < elems; ++i) {
    const float position_norm =
        std::sqrt((*h_Constview)[i].x() * (*h_Constview)[i].x() +
                  (*h_Constview)[i].y() * (*h_Constview)[i].y() +
                  (*h_Constview)[i].z() * (*h_Constview)[i].z());
    const double velocity_norm =
        std::sqrt((*h_Constview)[i].v_x() * (*h_Constview)[i].v_x() +
                  (*h_Constview)[i].v_y() * (*h_Constview)[i].v_y() +
                  (*h_Constview)[i].v_z() * (*h_Constview)[i].v_z());

    EXPECT_FLOAT_EQ(h_position_norms[i], position_norm);
    EXPECT_DOUBLE_EQ(h_velocity_norms[i], velocity_norm);
  }
}

TEST_F(SoACustomizedMethodsCudaTest, ViewMethodsCuda) {
  std::array<double, elems> times;
  times[0] = 0.0;
  for (std::size_t i = 1; i < elems; ++i) {
    times[i] = (*h_view)[i].x() / (*h_view)[i].v_x();
  }

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (elems + threadsPerBlock - 1) / threadsPerBlock;

  checkNormalise<<<blocksPerGrid, threadsPerBlock>>>(*d_view, d_times);
  CUDA_CHECK(cudaGetLastError());

  std::vector<double> h_times(elems);
  CUDA_CHECK(cudaMemcpy(h_times.data(), d_times, elems * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bufferSize, cudaMemcpyDeviceToHost));

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