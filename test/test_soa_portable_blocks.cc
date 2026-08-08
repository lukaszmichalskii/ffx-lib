#include <array>
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

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<>;

GENERATE_SOA_BLOCKS(SoAGenericBlocksTemplate, SOA_BLOCK(position, SoAPositionTemplate), SOA_BLOCK(pca, SoAPCATemplate))

using SoAGenericBlocks = SoAGenericBlocksTemplate<>;

// Test Fixture for Portable Collection SoA Blocks Tests
class SoAPortableBlocksTest : public ::testing::Test {
protected:
  static constexpr std::size_t elemsPos = 10;
  static constexpr std::size_t elemsPCA = 20;

  void SetUp() override {
    // Allocate Host Collections using host queue/device handle
    positionCollection = std::make_unique<ffx::soa::PortableHostCollection<SoAPosition>>(ffx::host(), elemsPos);
    pcaCollection = std::make_unique<ffx::soa::PortableHostCollection<SoAPCA>>(ffx::host(), elemsPCA);

    auto& posView = positionCollection->view();
    auto& pcaView = pcaCollection->view();

    // Fill position collection
    for (std::size_t i = 0; i < elemsPos; ++i) {
      posView[i] = {i * 1.0f, i * 2.0f, i * 3.0f};
    }
    posView.detectorType() = 1;

    // Fill PCA collection
    constexpr float time = 0.01f;
    for (std::size_t i = 0; i < elemsPCA; ++i) {
      pcaView[i].vector_1() = (i * 1.0f) / time;
      pcaView[i].vector_2() = (i * 2.0f) / time;
      pcaView[i].vector_3() = (i * 3.0f) / time;
      pcaView[i].candidateDirection()(0) = (i * 1.0) / time;
      pcaView[i].candidateDirection()(1) = (i * 2.0) / time;
      pcaView[i].candidateDirection()(2) = (i * 3.0) / time;
    }
  }

  std::unique_ptr<ffx::soa::PortableHostCollection<SoAPosition>> positionCollection;
  std::unique_ptr<ffx::soa::PortableHostCollection<SoAPCA>> pcaCollection;
};

TEST_F(SoAPortableBlocksTest, DeepCopyFromBlocksView) {
  auto& positionCollectionView = positionCollection->view();
  auto& pcaCollectionView = pcaCollection->view();

  // Construct a composite blocks view referencing independent collections
  SoAGenericBlocks::View genericBlocksView{positionCollectionView, pcaCollectionView};

  // Check equality of memory addresses (pointing to original collections)
  EXPECT_EQ(genericBlocksView.position().metadata().addressOf_x(), positionCollectionView.metadata().addressOf_x());
  EXPECT_EQ(genericBlocksView.position().metadata().addressOf_y(), positionCollectionView.metadata().addressOf_y());
  EXPECT_EQ(genericBlocksView.position().metadata().addressOf_z(), positionCollectionView.metadata().addressOf_z());
  EXPECT_EQ(genericBlocksView.pca().metadata().addressOf_candidateDirection(),
            pcaCollectionView.metadata().addressOf_candidateDirection());

  // PortableHostCollection holding aggregated contiguous block columns
  std::array<SoAGenericBlocks::size_type, 2> sizes{{elemsPos, elemsPCA}};
  ffx::soa::PortableHostCollection<SoAGenericBlocks> genericCollection(ffx::host(), sizes);
  genericCollection.deepCopy(genericBlocksView);

  // Check inequality of memory addresses (standalone buffer allocated)
  EXPECT_NE(genericCollection.view().position().metadata().addressOf_x(),
            positionCollectionView.metadata().addressOf_x());
  EXPECT_NE(genericCollection.view().position().metadata().addressOf_y(),
            positionCollectionView.metadata().addressOf_y());
  EXPECT_NE(genericCollection.view().position().metadata().addressOf_z(),
            positionCollectionView.metadata().addressOf_z());
  EXPECT_NE(genericCollection.view().pca().metadata().addressOf_candidateDirection(),
            pcaCollectionView.metadata().addressOf_candidateDirection());
}

TEST_F(SoAPortableBlocksTest, DeepCopyFromBlocksConstView) {
  const auto& positionCollectionConstView = positionCollection->const_view();
  const auto& pcaCollectionConstView = pcaCollection->const_view();

  // Construct a composite blocks const view referencing independent collections
  SoAGenericBlocks::ConstView genericBlocksConstView{positionCollectionConstView, pcaCollectionConstView};

  // Check equality of memory addresses (pointing to original collections)
  EXPECT_EQ(genericBlocksConstView.position().metadata().addressOf_x(),
            positionCollectionConstView.metadata().addressOf_x());
  EXPECT_EQ(genericBlocksConstView.position().metadata().addressOf_y(),
            positionCollectionConstView.metadata().addressOf_y());
  EXPECT_EQ(genericBlocksConstView.position().metadata().addressOf_z(),
            positionCollectionConstView.metadata().addressOf_z());
  EXPECT_EQ(genericBlocksConstView.pca().metadata().addressOf_candidateDirection(),
            pcaCollectionConstView.metadata().addressOf_candidateDirection());

  // PortableHostCollection holding aggregated contiguous block columns
  ffx::soa::PortableHostCollection<SoAGenericBlocks> genericCollection(ffx::host(), elemsPos, elemsPCA);
  genericCollection.deepCopy(genericBlocksConstView);

  // Check inequality of memory addresses (standalone buffer allocated)
  EXPECT_NE(genericCollection.const_view().position().metadata().addressOf_x(),
            positionCollectionConstView.metadata().addressOf_x());
  EXPECT_NE(genericCollection.const_view().position().metadata().addressOf_y(),
            positionCollectionConstView.metadata().addressOf_y());
  EXPECT_NE(genericCollection.const_view().position().metadata().addressOf_z(),
            positionCollectionConstView.metadata().addressOf_z());
  EXPECT_NE(genericCollection.const_view().pca().metadata().addressOf_candidateDirection(),
            pcaCollectionConstView.metadata().addressOf_candidateDirection());
}