#include <cstdint>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <alpaka/alpaka.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "ffx/ffx.h"

using namespace ffx_runtime;

using Vector5f = Eigen::Matrix<float, 5, 1>;
using Vector15f = Eigen::Matrix<float, 15, 1>;
using Matrix6x4d = Eigen::Matrix<double, 6, 4>;

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_COLUMN(float, quality),
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(std::int8_t, nLayers),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, pt),
                    SOA_EIGEN_COLUMN(Vector5f, state),
                    SOA_EIGEN_COLUMN(Vector15f, covariance),
                    SOA_EIGEN_COLUMN(Matrix6x4d, matrix),
                    SOA_SCALAR(int, nTracks),
                    SOA_COLUMN(std::uint32_t, hitOffsets))

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

namespace {

  template <typename F, std::size_t... Is>
  void unrollColumns(F&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
  }

  template <std::size_t N, typename F>
  void mergeSoAColumns(F&& f) {
    unrollColumns(std::forward<F>(f), std::make_index_sequence<N>{});
  }

  struct SumScalar {
    template <typename T>
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, T* result, const T* v1, const T* v2) const {
      *result = *v1 + *v2;
    }
  };

}  // namespace

TEST(SoAMergeAlpakaTest, ColumnMerge) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty()) {
    GTEST_SKIP() << "No devices available for the " << BOOST_PP_STRINGIZE(ffx_runtime) << " backend, skipping.";
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    constexpr int n1 = 10;
    constexpr int n2 = 20;

    ffx::soa::PortableHostCollection<SoA> hostCollection1(ffx::host(), n1);
    auto h_view1 = hostCollection1.view();

    ffx::soa::PortableHostCollection<SoA> hostCollection2(ffx::host(), n2);
    auto h_view2 = hostCollection2.view();

    // Fill first host collection
    for (int i = 0; i < hostCollection1.size(); ++i) {
      h_view1[i].quality() = static_cast<float>(1);
      h_view1[i].chi2() = static_cast<float>(2);
      h_view1[i].nLayers() = static_cast<std::int8_t>(3);
      h_view1[i].eta() = static_cast<float>(4);
      h_view1[i].pt() = static_cast<float>(5);
      h_view1[i].state().setConstant(6.0f);
      h_view1[i].covariance().setConstant(7.0f);
      h_view1[i].matrix().setConstant(8.0);
      h_view1[i].hitOffsets() = static_cast<std::uint32_t>(9);
    }
    h_view1.nTracks() = 8;

    // Fill second host collection
    for (int i = 0; i < hostCollection2.size(); ++i) {
      h_view2[i].quality() = static_cast<float>(11);
      h_view2[i].chi2() = static_cast<float>(12);
      h_view2[i].nLayers() = static_cast<std::int8_t>(13);
      h_view2[i].eta() = static_cast<float>(14);
      h_view2[i].pt() = static_cast<float>(15);
      h_view2[i].state().setConstant(16.0f);
      h_view2[i].covariance().setConstant(17.0f);
      h_view2[i].matrix().setConstant(18.0);
      h_view2[i].hitOffsets() = static_cast<std::uint32_t>(19);
    }
    h_view2.nTracks() = 17;

    ffx::soa::PortableCollection<Device, SoA> deviceCollection1(queue, hostCollection1.size());
    auto d_view1 = deviceCollection1.view();
    alpaka::memcpy(queue, deviceCollection1.buffer(), hostCollection1.buffer());

    ffx::soa::PortableCollection<Device, SoA> deviceCollection2(queue, hostCollection2.size());
    auto d_view2 = deviceCollection2.view();
    alpaka::memcpy(queue, deviceCollection2.buffer(), hostCollection2.buffer());

    const int nTk1 = h_view1.nTracks();
    const int nTk2 = h_view2.nTracks();
    const int nTotal = nTk1 + nTk2;

    ffx::soa::PortableCollection<Device, SoA> outputDevice(queue, nTk1 + nTk2);
    auto d_viewOut = outputDevice.view();

    alpaka::wait(queue);

    auto outDesc = SoA::Descriptor(d_viewOut);
    auto inDesc1 = SoA::ConstDescriptor(d_view1);
    auto inDesc2 = SoA::ConstDescriptor(d_view2);

    mergeSoAColumns<outDesc.num_cols>([&](auto columnIndex) {
      auto& outCol = std::get<columnIndex>(outDesc.buff);
      const auto& inCol1 = std::get<columnIndex>(inDesc1.buff);
      const auto& inCol2 = std::get<columnIndex>(inDesc2.buff);

      if constexpr (std::get<columnIndex>(outDesc.columnTypes) == ffx::soa::SoAColumnType::scalar) {
        alpaka::exec<Acc1D>(
            queue, ffx::make_workdiv<Acc1D>(1, 1), SumScalar{}, outCol.data(), inCol1.data(), inCol2.data());
      } else if constexpr (std::get<columnIndex>(outDesc.columnTypes) == ffx::soa::SoAColumnType::eigen) {
        using EigenType = typename std::tuple_element_t<columnIndex, decltype(outDesc.parameterTypes)>::ValueType;
        constexpr int nRows = EigenType::RowsAtCompileTime * EigenType::ColsAtCompileTime;

        const auto strideOutput = std::get<columnIndex>(outDesc.parameterTypes).stride();
        const auto strideInput1 = std::get<columnIndex>(inDesc1.parameterTypes).stride();
        const auto strideInput2 = std::get<columnIndex>(inDesc2.parameterTypes).stride();

        for (int i = 0; i < nRows; ++i) {
          const auto offsetOutput = i * strideOutput;
          const auto offsetIn1 = i * strideInput1;
          const auto offsetIn2 = i * strideInput2;
          alpaka::memcpy(queue,
                         ffx::make_device_view(queue, outCol.data() + offsetOutput, nTk1),
                         ffx::make_device_view(queue, inCol1.data() + offsetIn1, nTk1));
          // Copy second collection with offset of first collection size
          alpaka::memcpy(queue,
                         ffx::make_device_view(queue, outCol.data() + nTk1 + offsetOutput, nTk2),
                         ffx::make_device_view(queue, inCol2.data() + offsetIn2, nTk2));
        }
      } else {
        alpaka::memcpy(queue,
                       ffx::make_device_view(queue, outCol.data(), nTk1),
                       ffx::make_device_view(queue, inCol1.data(), nTk1));
        // Copy second collection with offset of first collection size
        alpaka::memcpy(queue,
                       ffx::make_device_view(queue, outCol.data() + nTk1, nTk2),
                       ffx::make_device_view(queue, inCol2.data(), nTk2));
      }
    });

    alpaka::wait(queue);

    ffx::soa::PortableHostCollection<SoA> outputHost(ffx::host(), nTk1 + nTk2);
    auto h_viewOut = outputHost.view();
    alpaka::memcpy(queue, outputHost.buffer(), outputDevice.buffer());

    alpaka::wait(queue);

    EXPECT_EQ(h_viewOut.nTracks(), nTotal);

    for (int i = 0; i < nTk1 + nTk2; ++i) {
      if (i < nTk1) {
        EXPECT_FLOAT_EQ(h_viewOut[i].quality(), h_view1[i].quality());
        EXPECT_FLOAT_EQ(h_viewOut[i].chi2(), h_view1[i].chi2());
        EXPECT_EQ(h_viewOut[i].nLayers(), h_view1[i].nLayers());
        EXPECT_FLOAT_EQ(h_viewOut[i].eta(), h_view1[i].eta());
        EXPECT_FLOAT_EQ(h_viewOut[i].pt(), h_view1[i].pt());
        EXPECT_TRUE(h_viewOut[i].state().isApprox(h_view1[i].state()));
        EXPECT_TRUE(h_viewOut[i].covariance().isApprox(h_view1[i].covariance()));
        EXPECT_TRUE(h_viewOut[i].matrix().isApprox(h_view1[i].matrix()));
      } else {
        const int idx2 = i - nTk1;
        EXPECT_FLOAT_EQ(h_viewOut[i].quality(), h_view2[idx2].quality());
        EXPECT_FLOAT_EQ(h_viewOut[i].chi2(), h_view2[idx2].chi2());
        EXPECT_EQ(h_viewOut[i].nLayers(), h_view2[idx2].nLayers());
        EXPECT_FLOAT_EQ(h_viewOut[i].eta(), h_view2[idx2].eta());
        EXPECT_FLOAT_EQ(h_viewOut[i].pt(), h_view2[idx2].pt());
        EXPECT_TRUE(h_viewOut[i].state().isApprox(h_view2[idx2].state()));
        EXPECT_TRUE(h_viewOut[i].covariance().isApprox(h_view2[idx2].covariance()));
        EXPECT_TRUE(h_viewOut[i].matrix().isApprox(h_view2[idx2].matrix()));
      }
    }
  }
}
