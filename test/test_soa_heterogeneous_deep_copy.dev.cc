#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <alpaka/alpaka.hpp>

#include "ffx/ffx.h"

using namespace ffx_runtime;

GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, eigenvalues),
                    SOA_COLUMN(float, eigenvector_1),
                    SOA_COLUMN(float, eigenvector_2),
                    SOA_COLUMN(float, eigenvector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;

GENERATE_SOA_LAYOUT(GenericSoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using GenericSoA = GenericSoATemplate<ffx::soa::CacheLineSize::Cpu>;
using GenericSoAView = GenericSoA::View;
using GenericSoAConstView = GenericSoA::ConstView;

// Kernel for filling the SoA
struct FillSoA {
  template <ffx::concepts::accelerator TAcc, typename PositionView, typename PCAView>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PositionView positionView, PCAView pcaView) const {
    constexpr float interval = 0.01f;
    if (alpaka::oncePerGrid(acc))
      positionView.detectorType() = 1;

    for (auto local_idx : alpaka::uniformElements(acc, positionView.metadata().size())) {
      positionView[local_idx].x() = static_cast<float>(local_idx);
      positionView[local_idx].y() = static_cast<float>(local_idx) * 2.0f;
      positionView[local_idx].z() = static_cast<float>(local_idx) * 3.0f;

      pcaView[local_idx].eigenvector_1() = positionView[local_idx].x() / interval;
      pcaView[local_idx].eigenvector_2() = positionView[local_idx].y() / interval;
      pcaView[local_idx].eigenvector_3() = positionView[local_idx].z() / interval;
      pcaView[local_idx].candidateDirection()(0) = positionView[local_idx].x() / interval;
      pcaView[local_idx].candidateDirection()(1) = positionView[local_idx].y() / interval;
      pcaView[local_idx].candidateDirection()(2) = positionView[local_idx].z() / interval;
    }
  }
};

class SoAPortableGenericViewAlpakaTest : public ::testing::Test {
protected:
  static constexpr std::size_t elems = 10;

  // Helper to initialize and fill collections so it can be reused per device
  void SetUpDeviceCollections(Queue& queue,
                              ffx::soa::PortableCollection<Device, SoAPosition>& positionCollection,
                              ffx::soa::PortableCollection<Device, SoAPCA>& pcaCollection) {
    auto positionCollectionView = positionCollection.view();
    auto pcaCollectionView = pcaCollection.view();

    auto blockSize = 64;
    auto numberOfBlocks = ffx::divide_up_by(elems, blockSize);
    const auto workDiv = ffx::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDiv, FillSoA{}, positionCollectionView, pcaCollectionView);
    alpaka::wait(queue);
  }
};

TEST_F(SoAPortableGenericViewAlpakaTest, DeepCopyViewHostToHostAndDeviceToDevice) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty()) {
    GTEST_SKIP() << "No devices available for the " << BOOST_PP_STRINGIZE(ffx_runtime) << " backend, skipping.";
  }

  for (auto const& device : devices) {
    Queue queue(device);

    ffx::soa::PortableCollection<Device, SoAPosition> positionCollection(queue, elems);
    ffx::soa::PortableCollection<Device, SoAPCA> pcaCollection(queue, elems);
    SetUpDeviceCollections(queue, positionCollection, pcaCollection);

    auto positionCollectionView = positionCollection.view();
    auto pcaCollectionView = pcaCollection.view();

    const auto posRecs = positionCollectionView.records();
    const auto pcaRecs = pcaCollectionView.records();

    GenericSoAView genericView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    EXPECT_EQ(genericView.metadata().addressOf_x(), positionCollectionView.metadata().addressOf_x());
    EXPECT_EQ(genericView.metadata().addressOf_y(), positionCollectionView.metadata().addressOf_y());
    EXPECT_EQ(genericView.metadata().addressOf_z(), positionCollectionView.metadata().addressOf_z());
    EXPECT_EQ(genericView.metadata().addressOf_candidateDirection(),
              pcaCollectionView.metadata().addressOf_candidateDirection());

    ffx::soa::PortableCollection<Device, GenericSoA> genericCollection(queue, elems);
    genericCollection.deepCopy(queue, genericView);

    EXPECT_NE(genericCollection.view().metadata().addressOf_x(), positionCollectionView.metadata().addressOf_x());
    EXPECT_NE(genericCollection.view().metadata().addressOf_y(), positionCollectionView.metadata().addressOf_y());
    EXPECT_NE(genericCollection.view().metadata().addressOf_z(), positionCollectionView.metadata().addressOf_z());
    EXPECT_NE(genericCollection.view().metadata().addressOf_candidateDirection(),
              pcaCollectionView.metadata().addressOf_candidateDirection());
  }
}

TEST_F(SoAPortableGenericViewAlpakaTest, DeepCopyConstViewHostToHostAndDeviceToDevice) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty())
    GTEST_SKIP();

  for (auto const& device : devices) {
    Queue queue(device);

    ffx::soa::PortableCollection<Device, SoAPosition> positionCollection(queue, elems);
    ffx::soa::PortableCollection<Device, SoAPCA> pcaCollection(queue, elems);
    SetUpDeviceCollections(queue, positionCollection, pcaCollection);

    const auto positionCollectionConstView = positionCollection.const_view();
    const auto pcaCollectionConstView = pcaCollection.const_view();

    const auto posRecs = positionCollectionConstView.records();
    const auto pcaRecs = pcaCollectionConstView.records();

    GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    EXPECT_EQ(genericConstView.metadata().addressOf_x(), positionCollectionConstView.metadata().addressOf_x());
    EXPECT_EQ(genericConstView.metadata().addressOf_y(), positionCollectionConstView.metadata().addressOf_y());
    EXPECT_EQ(genericConstView.metadata().addressOf_z(), positionCollectionConstView.metadata().addressOf_z());
    EXPECT_EQ(genericConstView.metadata().addressOf_candidateDirection(),
              pcaCollectionConstView.metadata().addressOf_candidateDirection());

    ffx::soa::PortableCollection<Device, GenericSoA> genericCollection(queue, elems);
    genericCollection.deepCopy(queue, genericConstView);

    EXPECT_NE(genericCollection.view().metadata().addressOf_x(), positionCollectionConstView.metadata().addressOf_x());
    EXPECT_NE(genericCollection.view().metadata().addressOf_y(), positionCollectionConstView.metadata().addressOf_y());
    EXPECT_NE(genericCollection.view().metadata().addressOf_z(), positionCollectionConstView.metadata().addressOf_z());
    EXPECT_NE(genericCollection.view().metadata().addressOf_candidateDirection(),
              pcaCollectionConstView.metadata().addressOf_candidateDirection());

    // Check correctness of the copy
    ffx::soa::PortableHostCollection<GenericSoA> genericHostCollection(queue, elems);
    ffx::soa::PortableHostCollection<SoAPosition> positionHostCollection(queue, elems);
    ffx::soa::PortableHostCollection<SoAPCA> pcaHostCollection(queue, elems);

    alpaka::memcpy(queue, genericHostCollection.buffer(), genericCollection.buffer());
    alpaka::memcpy(queue, positionHostCollection.buffer(), positionCollection.buffer());
    alpaka::memcpy(queue, pcaHostCollection.buffer(), pcaCollection.buffer());
    alpaka::wait(queue);

    const auto genericViewHostCollection = genericHostCollection.const_view();
    const auto positionViewHostCollection = positionHostCollection.const_view();
    const auto pcaViewHostCollection = pcaHostCollection.const_view();

    for (std::size_t i = 0; i < elems; ++i) {
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].x(), positionViewHostCollection[i].x());
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].y(), positionViewHostCollection[i].y());
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].z(), positionViewHostCollection[i].z());
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].candidateDirection()(0),
                      pcaViewHostCollection[i].candidateDirection()(0));
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].candidateDirection()(1),
                      pcaViewHostCollection[i].candidateDirection()(1));
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].candidateDirection()(2),
                      pcaViewHostCollection[i].candidateDirection()(2));
    }
  }
}

TEST_F(SoAPortableGenericViewAlpakaTest, DeepCopyConstViewDeviceToHost) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty())
    GTEST_SKIP();

  for (auto const& device : devices) {
    Queue queue(device);

    ffx::soa::PortableCollection<Device, SoAPosition> positionCollection(queue, elems);
    ffx::soa::PortableCollection<Device, SoAPCA> pcaCollection(queue, elems);
    SetUpDeviceCollections(queue, positionCollection, pcaCollection);

    const auto positionCollectionConstView = positionCollection.const_view();
    const auto pcaCollectionConstView = pcaCollection.const_view();

    const auto posRecs = positionCollectionConstView.records();
    const auto pcaRecs = pcaCollectionConstView.records();

    GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    EXPECT_EQ(genericConstView.metadata().addressOf_x(), positionCollectionConstView.metadata().addressOf_x());
    EXPECT_EQ(genericConstView.metadata().addressOf_y(), positionCollectionConstView.metadata().addressOf_y());
    EXPECT_EQ(genericConstView.metadata().addressOf_z(), positionCollectionConstView.metadata().addressOf_z());
    EXPECT_EQ(genericConstView.metadata().addressOf_candidateDirection(),
              pcaCollectionConstView.metadata().addressOf_candidateDirection());

    ffx::soa::PortableHostCollection<GenericSoA> genericCollection(queue, elems);
    genericCollection.deepCopy(queue, genericConstView);

    EXPECT_NE(genericCollection.view().metadata().addressOf_x(), positionCollectionConstView.metadata().addressOf_x());
    EXPECT_NE(genericCollection.view().metadata().addressOf_y(), positionCollectionConstView.metadata().addressOf_y());
    EXPECT_NE(genericCollection.view().metadata().addressOf_z(), positionCollectionConstView.metadata().addressOf_z());
    EXPECT_NE(genericCollection.view().metadata().addressOf_candidateDirection(),
              pcaCollectionConstView.metadata().addressOf_candidateDirection());

    // Check correctness of the copy
    ffx::soa::PortableHostCollection<SoAPosition> positionHostCollection(queue, elems);
    ffx::soa::PortableHostCollection<SoAPCA> pcaHostCollection(queue, elems);

    alpaka::memcpy(queue, positionHostCollection.buffer(), positionCollection.buffer());
    alpaka::memcpy(queue, pcaHostCollection.buffer(), pcaCollection.buffer());
    alpaka::wait(queue);

    const auto genericViewCollection = genericCollection.const_view();
    const auto positionViewHostCollection = positionHostCollection.const_view();
    const auto pcaViewHostCollection = pcaHostCollection.const_view();

    for (std::size_t i = 0; i < elems; ++i) {
      EXPECT_FLOAT_EQ(genericViewCollection[i].x(), positionViewHostCollection[i].x());
      EXPECT_FLOAT_EQ(genericViewCollection[i].y(), positionViewHostCollection[i].y());
      EXPECT_FLOAT_EQ(genericViewCollection[i].z(), positionViewHostCollection[i].z());
      EXPECT_FLOAT_EQ(genericViewCollection[i].candidateDirection()(0),
                      pcaViewHostCollection[i].candidateDirection()(0));
      EXPECT_FLOAT_EQ(genericViewCollection[i].candidateDirection()(1),
                      pcaViewHostCollection[i].candidateDirection()(1));
      EXPECT_FLOAT_EQ(genericViewCollection[i].candidateDirection()(2),
                      pcaViewHostCollection[i].candidateDirection()(2));
    }
  }
}

TEST_F(SoAPortableGenericViewAlpakaTest, DeepCopyConstViewHostToDevice) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty())
    GTEST_SKIP();

  for (auto const& device : devices) {
    Queue queue(device);

    ffx::soa::PortableCollection<Device, SoAPosition> positionCollection(queue, elems);
    ffx::soa::PortableCollection<Device, SoAPCA> pcaCollection(queue, elems);
    SetUpDeviceCollections(queue, positionCollection, pcaCollection);

    ffx::soa::PortableHostCollection<SoAPosition> positionHostCollection(queue, elems);
    ffx::soa::PortableHostCollection<SoAPCA> pcaHostCollection(queue, elems);

    alpaka::memcpy(queue, positionHostCollection.buffer(), positionCollection.buffer());
    alpaka::memcpy(queue, pcaHostCollection.buffer(), pcaCollection.buffer());

    const auto positionViewHostCollection = positionHostCollection.const_view();
    const auto pcaViewHostCollection = pcaHostCollection.const_view();

    const auto posRecs = positionViewHostCollection.records();
    const auto pcaRecs = pcaViewHostCollection.records();

    GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

    EXPECT_EQ(genericConstView.metadata().addressOf_x(), positionViewHostCollection.metadata().addressOf_x());
    EXPECT_EQ(genericConstView.metadata().addressOf_y(), positionViewHostCollection.metadata().addressOf_y());
    EXPECT_EQ(genericConstView.metadata().addressOf_z(), positionViewHostCollection.metadata().addressOf_z());
    EXPECT_EQ(genericConstView.metadata().addressOf_candidateDirection(),
              pcaViewHostCollection.metadata().addressOf_candidateDirection());

    ffx::soa::PortableCollection<Device, GenericSoA> genericCollection(queue, elems);
    genericCollection.deepCopy(queue, genericConstView);

    EXPECT_NE(genericCollection.view().metadata().addressOf_x(), positionViewHostCollection.metadata().addressOf_x());
    EXPECT_NE(genericCollection.view().metadata().addressOf_y(), positionViewHostCollection.metadata().addressOf_y());
    EXPECT_NE(genericCollection.view().metadata().addressOf_z(), positionViewHostCollection.metadata().addressOf_z());
    EXPECT_NE(genericCollection.view().metadata().addressOf_candidateDirection(),
              pcaViewHostCollection.metadata().addressOf_candidateDirection());

    // Check correctness of the copy
    ffx::soa::PortableHostCollection<GenericSoA> genericHostCollection(queue, elems);
    alpaka::memcpy(queue, genericHostCollection.buffer(), genericCollection.buffer());
    alpaka::wait(queue);

    const auto genericViewHostCollection = genericHostCollection.const_view();

    for (std::size_t i = 0; i < elems; ++i) {
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].x(), positionViewHostCollection[i].x());
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].y(), positionViewHostCollection[i].y());
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].z(), positionViewHostCollection[i].z());
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].candidateDirection()(0),
                      pcaViewHostCollection[i].candidateDirection()(0));
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].candidateDirection()(1),
                      pcaViewHostCollection[i].candidateDirection()(1));
      EXPECT_FLOAT_EQ(genericViewHostCollection[i].candidateDirection()(2),
                      pcaViewHostCollection[i].candidateDirection()(2));
    }
  }
}