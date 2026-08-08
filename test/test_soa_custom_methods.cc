#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include <gtest/gtest.h>

#include "test_soa_definition_custom_methods.h"

// Test Fixture for Host CPU Customized Methods Tests
class SoACustomizedMethodsTest : public ::testing::Test {
protected:
  static constexpr std::size_t elems = 10;

  void SetUp() override {
    bufferSize = SoA::computeDataSize(elems);
    buffer.reset(reinterpret_cast<std::byte*>(aligned_alloc(SoA::alignment, bufferSize)));
    ASSERT_NE(buffer, nullptr);

    soa = std::make_unique<SoA>(buffer.get(), elems);
    view = std::make_unique<SoAView>(*soa);
    const_view = std::make_unique<SoAConstView>(*soa);

    // Populate data
    for (std::size_t i = 0; i < elems; ++i) {
      (*view)[i].x() = static_cast<float>(i);
      (*view)[i].y() = static_cast<float>(i) * 2.0f;
      (*view)[i].z() = static_cast<float>(i) * 3.0f;
      (*view)[i].v_x() = static_cast<double>(i);
      (*view)[i].v_y() = static_cast<double>(i) * 20.0;
      (*view)[i].v_z() = static_cast<double>(i) * 30.0;
    }
    view->detectorType() = 42;
  }

  std::size_t bufferSize{0};
  std::unique_ptr<std::byte, decltype(&std::free)> buffer{nullptr, std::free};
  std::unique_ptr<SoA> soa;
  std::unique_ptr<SoAView> view;
  std::unique_ptr<SoAConstView> const_view;
};

TEST_F(SoACustomizedMethodsTest, ConstViewMethods) {
  std::array<float, elems> position_norms;
  std::array<double, elems> velocity_norms;

  // Check correctness of square_norm() functions
  for (std::size_t i = 0; i < elems; ++i) {
    position_norms[i] =
        std::sqrt((*const_view)[i].x() * (*const_view)[i].x() + (*const_view)[i].y() * (*const_view)[i].y() +
                  (*const_view)[i].z() * (*const_view)[i].z());
    velocity_norms[i] =
        std::sqrt((*const_view)[i].v_x() * (*const_view)[i].v_x() + (*const_view)[i].v_y() * (*const_view)[i].v_y() +
                  (*const_view)[i].v_z() * (*const_view)[i].v_z());

    EXPECT_FLOAT_EQ(position_norms[i], (*const_view)[i].square_norm_position());
    EXPECT_DOUBLE_EQ(velocity_norms[i], (*const_view)[i].square_norm_velocity());
  }
}

TEST_F(SoACustomizedMethodsTest, ViewMethods) {
  std::array<double, elems> times;

  // Check correctness of time() function
  times[0] = 0.0;
  for (std::size_t i = 0; i < elems; ++i) {
    if (i != 0) {
      times[i] = (*view)[i].x() / (*view)[i].v_x();
    }
    EXPECT_DOUBLE_EQ(times[i], SoAView::const_element::time((*view)[i].x(), (*view)[i].v_x()));
  }

  // Normalize particles data
  for (std::size_t i = 0; i < elems; ++i) {
    (*view)[i].normalise();
  }

  // Check norm equal to 1 (or 0 for index 0)
  EXPECT_FLOAT_EQ((*view)[0].square_norm_position(), 0.0f);
  EXPECT_DOUBLE_EQ((*view)[0].square_norm_velocity(), 0.0);

  for (std::size_t i = 1; i < elems; ++i) {
    EXPECT_NEAR((*view)[i].square_norm_position(), 1.0f, 1e-6f);
    EXPECT_NEAR((*view)[i].square_norm_velocity(), 1.0, 1e-9);
  }
}