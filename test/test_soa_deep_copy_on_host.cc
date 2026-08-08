#include <cstddef>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "ffx/ffx.h"

// Define SoA Layouts
GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;

GENERATE_SOA_LAYOUT(GenericSoATemplate,
                    SOA_COLUMN(float, xPos),
                    SOA_COLUMN(float, yPos),
                    SOA_COLUMN(float, zPos),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using GenericSoA = GenericSoATemplate<>;
using GenericSoAView = GenericSoA::View;
using GenericSoAConstView = GenericSoA::ConstView;

// Test Fixture for Portable Collection Generic View Tests
class SoAPortableGenericViewTest : public ::testing::Test {
protected:
  static constexpr std::size_t elems = 10;

  void SetUp() override {
    // Allocate Host Collections using host queue/device handle
    positionCollection = std::make_unique<ffx::soa::PortableHostCollection<SoAPosition>>(ffx::host(), elems);
    pcaCollection = std::make_unique<ffx::soa::PortableHostCollection<SoAPCA>>(ffx::host(), elems);

    auto& posView = positionCollection->view();
    auto& pcaView = pcaCollection->view();

    // Fill position collection
    for (std::size_t i = 0; i < elems; ++i) {
      posView[i] = {i * 1.0f, i * 2.0f, i * 3.0f};
    }
    posView.detectorType() = 1;

    // Fill PCA collection using data from position collection
    constexpr float time = 0.01f;
    for (std::size_t i = 0; i < elems; ++i) {
      pcaView[i].vector_1() = posView[i].x() / time;
      pcaView[i].vector_2() = posView[i].y() / time;
      pcaView[i].vector_3() = posView[i].z() / time;
      pcaView[i].candidateDirection()(0) = posView[i].x() / time;
      pcaView[i].candidateDirection()(1) = posView[i].y() / time;
      pcaView[i].candidateDirection()(2) = posView[i].z() / time;
    }
  }

  std::unique_ptr<ffx::soa::PortableHostCollection<SoAPosition>> positionCollection;
  std::unique_ptr<ffx::soa::PortableHostCollection<SoAPCA>> pcaCollection;
};

TEST_F(SoAPortableGenericViewTest, DeepCopyFromGenericView) {
  auto& positionCollectionView = positionCollection->view();
  auto& pcaCollectionView = pcaCollection->view();

  // Obtain records containing column pointers
  const auto posRecs = positionCollectionView.records();
  const auto pcaRecs = pcaCollectionView.records();

  // Construct a GenericSoAView across individual collection column pointers
  GenericSoAView genericView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

  // Check equality of memory addresses (referencing original collections)
  EXPECT_EQ(genericView.metadata().addressOf_xPos(), positionCollectionView.metadata().addressOf_x());
  EXPECT_EQ(genericView.metadata().addressOf_yPos(), positionCollectionView.metadata().addressOf_y());
  EXPECT_EQ(genericView.metadata().addressOf_zPos(), positionCollectionView.metadata().addressOf_z());
  EXPECT_EQ(genericView.metadata().addressOf_candidateDirection(),
            pcaCollectionView.metadata().addressOf_candidateDirection());

  // PortableHostCollection hosting the aggregated contiguous columns
  ffx::soa::PortableHostCollection<GenericSoA> genericCollection(ffx::host(), elems);
  genericCollection.deepCopy(genericView);

  // Check inequality of memory addresses (pointing to deep-copied memory buffer)
  EXPECT_NE(genericCollection.view().metadata().addressOf_xPos(), positionCollectionView.metadata().addressOf_x());
  EXPECT_NE(genericCollection.view().metadata().addressOf_yPos(), positionCollectionView.metadata().addressOf_y());
  EXPECT_NE(genericCollection.view().metadata().addressOf_zPos(), positionCollectionView.metadata().addressOf_z());
  EXPECT_NE(genericCollection.view().metadata().addressOf_candidateDirection(),
            pcaCollectionView.metadata().addressOf_candidateDirection());
}

TEST_F(SoAPortableGenericViewTest, DeepCopyFromGenericConstView) {
  const auto& positionCollectionConstView = positionCollection->const_view();
  const auto& pcaCollectionConstView = pcaCollection->const_view();

  const auto posRecs = positionCollectionConstView.records();
  const auto pcaRecs = pcaCollectionConstView.records();

  // Construct a GenericSoAConstView across individual collection column pointers
  GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

  // Check equality of memory addresses (referencing original collections)
  EXPECT_EQ(genericConstView.metadata().addressOf_xPos(), positionCollectionConstView.metadata().addressOf_x());
  EXPECT_EQ(genericConstView.metadata().addressOf_yPos(), positionCollectionConstView.metadata().addressOf_y());
  EXPECT_EQ(genericConstView.metadata().addressOf_zPos(), positionCollectionConstView.metadata().addressOf_z());
  EXPECT_EQ(genericConstView.metadata().addressOf_candidateDirection(),
            pcaCollectionConstView.metadata().addressOf_candidateDirection());

  // PortableHostCollection hosting the aggregated contiguous columns
  ffx::soa::PortableHostCollection<GenericSoA> genericCollection(ffx::host(), elems);
  genericCollection.deepCopy(genericConstView);

  // Check inequality of memory addresses (pointing to deep-copied memory buffer)
  EXPECT_NE(genericCollection.const_view().metadata().addressOf_xPos(),
            positionCollectionConstView.metadata().addressOf_x());
  EXPECT_NE(genericCollection.const_view().metadata().addressOf_yPos(),
            positionCollectionConstView.metadata().addressOf_y());
  EXPECT_NE(genericCollection.const_view().metadata().addressOf_zPos(),
            positionCollectionConstView.metadata().addressOf_z());
  EXPECT_NE(genericCollection.const_view().metadata().addressOf_candidateDirection(),
            pcaCollectionConstView.metadata().addressOf_candidateDirection());
}