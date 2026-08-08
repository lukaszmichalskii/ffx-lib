#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>
#include <alpaka/alpaka.hpp>

#include "ffx/ffx.h"

using namespace ffx_runtime;

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(double, v_x),
                    SOA_COLUMN(double, v_y),
                    SOA_COLUMN(double, v_z),

                    SOA_ELEMENT_METHODS(SOA_HOST_DEVICE void normalise() {
                      float norm_position = square_norm_position();
                      if (norm_position > 0.0f) {
                        x() /= norm_position;
                        y() /= norm_position;
                        z() /= norm_position;
                      }
                      double norm_velocity = square_norm_velocity();
                      if (norm_velocity > 0.0f) {
                        v_x() /= norm_velocity;
                        v_y() /= norm_velocity;
                        v_z() /= norm_velocity;
                      }
                    }),

                    SOA_CONST_ELEMENT_METHODS(
                        SOA_HOST_DEVICE float square_norm_position()
                            const { return std::sqrt(x() * x() + y() * y() + z() * z()); }

                        SOA_HOST_DEVICE double square_norm_velocity()
                            const { return std::sqrt(v_x() * v_x() + v_y() * v_y() + v_z() * v_z()); }

                        template <typename T1, typename T2>
                        SOA_HOST_DEVICE static auto time(T1 pos, T2 vel) {
                          if (vel != 0)
                            return pos / vel;
                          return 0.0;
                        }),

                    SOA_SCALAR(int, detectorType))

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

GENERATE_SOA_LAYOUT(ResultTemplate,
                    SOA_COLUMN(float, positionNorm),
                    SOA_COLUMN(double, velocityNorm),
                    SOA_COLUMN(double, times))

using ResultSoA = ResultTemplate<>;
using ResultView = ResultSoA::View;

struct calculateNorm {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, SoAConstView soaConstView, ResultView resultView) const {
    for (auto i : alpaka::uniformElements(acc, soaConstView.metadata().size())) {
      resultView[i].positionNorm() = soaConstView[i].square_norm_position();
      resultView[i].velocityNorm() = soaConstView[i].square_norm_velocity();
    }
  }
};

struct checkNormalise {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, SoAView soaView, ResultView resultView) const {
    for (auto i : alpaka::uniformElements(acc, soaView.metadata().size())) {
      resultView[i].times() = SoAView::const_element::time(soaView[i].x(), soaView[i].v_x());
      soaView[i].normalise();
    }
  }
};

class SoACustomizedMethodsAlpakaTest : public ::testing::Test {
protected:
  static constexpr std::size_t elems = 10;
};

TEST_F(SoACustomizedMethodsAlpakaTest, ConstViewMethodsAlpaka) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty()) {
    GTEST_SKIP() << "No devices available for the " << BOOST_PP_STRINGIZE(ffx_runtime) << " backend, skipping.";
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    ffx::soa::PortableHostCollection<SoA> hostCollection(ffx::host(), elems);
    auto h_view = hostCollection.view();
    const auto h_Constview = hostCollection.const_view();

    ffx::soa::PortableHostCollection<ResultSoA> hostResultCollection(ffx::host(), elems);
    auto h_result_view = hostResultCollection.view();

    // Fill host collection
    for (std::size_t i = 0; i < elems; ++i) {
      h_view[i].x() = static_cast<float>(i);
      h_view[i].y() = static_cast<float>(i) * 2.0f;
      h_view[i].z() = static_cast<float>(i) * 3.0f;
      h_view[i].v_x() = static_cast<double>(i);
      h_view[i].v_y() = static_cast<double>(i) * 20.0;
      h_view[i].v_z() = static_cast<double>(i) * 30.0;
    }
    h_view.detectorType() = 42;

    ffx::soa::PortableCollection<Device, SoA> deviceCollection(queue, elems);
    auto d_Constview = deviceCollection.const_view();
    alpaka::memcpy(queue, deviceCollection.buffer(), hostCollection.buffer());

    ffx::soa::PortableCollection<Device, ResultSoA> deviceResultCollection(queue, elems);
    auto d_result_view = deviceResultCollection.view();
    alpaka::wait(queue);

    // Work division setup
    const std::size_t blockSize = 256;
    const std::size_t numberOfBlocks = ffx::divide_up_by(elems, blockSize);
    const auto workDiv = ffx::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDiv, calculateNorm{}, d_Constview, d_result_view);
    alpaka::memcpy(queue, hostResultCollection.buffer(), deviceResultCollection.buffer());
    alpaka::wait(queue);

    // Check correctness of square_norm() functions
    for (std::size_t i = 0; i < elems; ++i) {
      const float position_norm =
          std::sqrt(h_Constview[i].x() * h_Constview[i].x() + h_Constview[i].y() * h_Constview[i].y() +
                    h_Constview[i].z() * h_Constview[i].z());
      const double velocity_norm =
          std::sqrt(h_Constview[i].v_x() * h_Constview[i].v_x() + h_Constview[i].v_y() * h_Constview[i].v_y() +
                    h_Constview[i].v_z() * h_Constview[i].v_z());

      EXPECT_FLOAT_EQ(h_result_view[i].positionNorm(), position_norm);
      EXPECT_DOUBLE_EQ(h_result_view[i].velocityNorm(), velocity_norm);
    }
  }
}

TEST_F(SoACustomizedMethodsAlpakaTest, ViewMethodsAlpaka) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty()) {
    GTEST_SKIP() << "No devices available for the " << BOOST_PP_STRINGIZE(ffx_runtime) << " backend, skipping.";
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    ffx::soa::PortableHostCollection<SoA> hostCollection(ffx::host(), elems);
    auto h_view = hostCollection.view();

    ffx::soa::PortableHostCollection<ResultSoA> hostResultCollection(ffx::host(), elems);
    auto h_result_view = hostResultCollection.view();

    // Fill host collection
    for (std::size_t i = 0; i < elems; ++i) {
      h_view[i].x() = static_cast<float>(i);
      h_view[i].y() = static_cast<float>(i) * 2.0f;
      h_view[i].z() = static_cast<float>(i) * 3.0f;
      h_view[i].v_x() = static_cast<double>(i);
      h_view[i].v_y() = static_cast<double>(i) * 20.0;
      h_view[i].v_z() = static_cast<double>(i) * 30.0;
    }
    h_view.detectorType() = 42;

    ffx::soa::PortableCollection<Device, SoA> deviceCollection(queue, elems);
    auto d_view = deviceCollection.view();
    alpaka::memcpy(queue, deviceCollection.buffer(), hostCollection.buffer());

    ffx::soa::PortableCollection<Device, ResultSoA> deviceResultCollection(queue, elems);
    auto d_result_view = deviceResultCollection.view();
    alpaka::wait(queue);

    std::array<double, elems> times;
    times[0] = 0.0;
    for (std::size_t i = 1; i < elems; ++i) {
      times[i] = h_view[i].x() / h_view[i].v_x();
    }

    const std::size_t blockSize = 256;
    const std::size_t numberOfBlocks = ffx::divide_up_by(elems, blockSize);
    const auto workDiv = ffx::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDiv, checkNormalise{}, d_view, d_result_view);
    alpaka::memcpy(queue, hostResultCollection.buffer(), deviceResultCollection.buffer());
    alpaka::memcpy(queue, hostCollection.buffer(), deviceCollection.buffer());
    alpaka::wait(queue);

    // Check correctness of time() function
    for (std::size_t i = 0; i < elems; ++i) {
      EXPECT_DOUBLE_EQ(h_result_view[i].times(), times[i]);
    }

    EXPECT_FLOAT_EQ(h_view[0].square_norm_position(), 0.0f);
    EXPECT_DOUBLE_EQ(h_view[0].square_norm_velocity(), 0.0);

    for (std::size_t i = 1; i < elems; ++i) {
      EXPECT_NEAR(h_view[i].square_norm_position(), 1.0f, 1e-6f);
      EXPECT_NEAR(h_view[i].square_norm_velocity(), 1.0, 1e-9);
    }
  }
}