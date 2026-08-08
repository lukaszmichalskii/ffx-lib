#include <cstdlib>
#include <memory>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "ffx/ffx.h"

constexpr float step = 0.01f;

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

GENERATE_SOA_BLOCKS(SoAGenericBlocksTemplate, SOA_BLOCK(position, SoAPositionTemplate), SOA_BLOCK(pca, SoAPCATemplate))

using SoAGenericBlocks = SoAGenericBlocksTemplate<>;
using SoAGenericBlocksView = SoAGenericBlocks::View;
using SoAGenericBlocksConstView = SoAGenericBlocks::ConstView;

// Test Fixture for Generic Blocks Tests
class SoAGenericBlocksTest : public ::testing::Test {
protected:
  static constexpr std::size_t elemsPos = 10;
  static constexpr std::size_t elemsPCA = 20;

  void SetUp() override {
    const std::size_t positionBufferSize = SoAPosition::computeDataSize(elemsPos);
    const std::size_t pcaBufferSize = SoAPCA::computeDataSize(elemsPCA);

    bufferPos.reset(reinterpret_cast<std::byte*>(aligned_alloc(SoAPosition::alignment, positionBufferSize)));
    bufferPCA.reset(reinterpret_cast<std::byte*>(aligned_alloc(SoAPCA::alignment, pcaBufferSize)));

    position = std::make_unique<SoAPosition>(bufferPos.get(), elemsPos);
    pca = std::make_unique<SoAPCA>(bufferPCA.get(), elemsPCA);

    positionView = std::make_unique<SoAPositionView>(*position);
    positionConstView = std::make_unique<SoAPositionConstView>(*position);
    pcaView = std::make_unique<SoAPCAView>(*pca);
    pcaConstView = std::make_unique<SoAPCAConstView>(*pca);

    // Fill up initial data
    for (std::size_t i = 0; i < elemsPos; ++i) {
      (*positionView)[i] = {i * 1.0f, i * 2.0f, i * 3.0f};
    }
    positionView->detectorType() = 1;

    for (std::size_t i = 0; i < elemsPCA; ++i) {
      (*pcaView)[i].vector_1() = (i * 1.0f) / step;
      (*pcaView)[i].vector_2() = (i * 2.0f) / step;
      (*pcaView)[i].vector_3() = (i * 3.0f) / step;
      (*pcaView)[i].candidateDirection()(0) = (i * 1.0) / step;
      (*pcaView)[i].candidateDirection()(1) = (i * 2.0) / step;
      (*pcaView)[i].candidateDirection()(2) = (i * 3.0) / step;
    }
  }

  std::unique_ptr<std::byte, decltype(&std::free)> bufferPos{nullptr, std::free};
  std::unique_ptr<std::byte, decltype(&std::free)> bufferPCA{nullptr, std::free};

  std::unique_ptr<SoAPosition> position;
  std::unique_ptr<SoAPCA> pca;

  std::unique_ptr<SoAPositionView> positionView;
  std::unique_ptr<SoAPositionConstView> positionConstView;
  std::unique_ptr<SoAPCAView> pcaView;
  std::unique_ptr<SoAPCAConstView> pcaConstView;
};

TEST_F(SoAGenericBlocksTest, GenericBlocksView) {
  // Construct View from distinct block views
  SoAGenericBlocksView genericBlocksView{*positionView, *pcaView};

  // Verify metadata
  EXPECT_EQ(genericBlocksView.metadata().size()[0], elemsPos);
  EXPECT_EQ(genericBlocksView.position().metadata().size(), elemsPos);
  EXPECT_EQ(genericBlocksView.metadata().size()[1], elemsPCA);
  EXPECT_EQ(genericBlocksView.pca().metadata().size(), elemsPCA);

  // Check equality of memory addresses
  EXPECT_EQ(genericBlocksView.position().metadata().addressOf_x(), positionView->metadata().addressOf_x());
  EXPECT_EQ(genericBlocksView.position().metadata().addressOf_y(), positionView->metadata().addressOf_y());
  EXPECT_EQ(genericBlocksView.position().metadata().addressOf_z(), positionView->metadata().addressOf_z());
  EXPECT_EQ(genericBlocksView.pca().metadata().addressOf_candidateDirection(),
            pcaView->metadata().addressOf_candidateDirection());

  // Verify data
  for (std::size_t i = 0; i < genericBlocksView.position().metadata().size(); ++i) {
    auto pos = genericBlocksView.position()[i];
    EXPECT_FLOAT_EQ(pos.x(), static_cast<float>(i));
    EXPECT_FLOAT_EQ(pos.y(), static_cast<float>(i * 2.0f));
    EXPECT_FLOAT_EQ(pos.z(), static_cast<float>(i * 3.0f));
  }

  for (std::size_t i = 0; i < genericBlocksView.pca().metadata().size(); ++i) {
    auto pca_val = genericBlocksView.pca()[i];
    EXPECT_FLOAT_EQ(pca_val.vector_1(), static_cast<float>(i) / step);
    EXPECT_FLOAT_EQ(pca_val.vector_2(), static_cast<float>(i * 2.0f) / step);
    EXPECT_FLOAT_EQ(pca_val.vector_3(), static_cast<float>(i * 3.0f) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(0), static_cast<double>(i) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(1), static_cast<double>(i * 2.0) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(2), static_cast<double>(i * 3.0) / step);
  }
}

TEST_F(SoAGenericBlocksTest, GenericBlocksConstView) {
  SoAGenericBlocksConstView genericBlocksConstView{*positionConstView, *pcaConstView};

  // Verify metadata
  EXPECT_EQ(genericBlocksConstView.metadata().size()[0], elemsPos);
  EXPECT_EQ(genericBlocksConstView.position().metadata().size(), elemsPos);
  EXPECT_EQ(genericBlocksConstView.metadata().size()[1], elemsPCA);
  EXPECT_EQ(genericBlocksConstView.pca().metadata().size(), elemsPCA);

  // Check equality of memory addresses
  EXPECT_EQ(genericBlocksConstView.position().metadata().addressOf_x(), positionConstView->metadata().addressOf_x());
  EXPECT_EQ(genericBlocksConstView.position().metadata().addressOf_y(), positionConstView->metadata().addressOf_y());
  EXPECT_EQ(genericBlocksConstView.position().metadata().addressOf_z(), positionConstView->metadata().addressOf_z());
  EXPECT_EQ(genericBlocksConstView.pca().metadata().addressOf_candidateDirection(),
            pcaConstView->metadata().addressOf_candidateDirection());

  // Verify data
  for (std::size_t i = 0; i < genericBlocksConstView.position().metadata().size(); ++i) {
    auto pos = genericBlocksConstView.position()[i];
    EXPECT_FLOAT_EQ(pos.x(), static_cast<float>(i));
    EXPECT_FLOAT_EQ(pos.y(), static_cast<float>(i * 2.0f));
    EXPECT_FLOAT_EQ(pos.z(), static_cast<float>(i * 3.0f));
  }

  for (std::size_t i = 0; i < genericBlocksConstView.pca().metadata().size(); ++i) {
    auto pca_val = genericBlocksConstView.pca()[i];
    EXPECT_FLOAT_EQ(pca_val.vector_1(), static_cast<float>(i) / step);
    EXPECT_FLOAT_EQ(pca_val.vector_2(), static_cast<float>(i * 2.0f) / step);
    EXPECT_FLOAT_EQ(pca_val.vector_3(), static_cast<float>(i * 3.0f) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(0), static_cast<double>(i) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(1), static_cast<double>(i * 2.0) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(2), static_cast<double>(i * 3.0) / step);
  }
}

TEST_F(SoAGenericBlocksTest, GenericBlocksConstViewFromMutableViews) {
  SoAGenericBlocksConstView genericBlocksConstView{*positionView, *pcaView};

  // Verify data
  for (std::size_t i = 0; i < genericBlocksConstView.position().metadata().size(); ++i) {
    auto pos = genericBlocksConstView.position()[i];
    EXPECT_FLOAT_EQ(pos.x(), static_cast<float>(i));
    EXPECT_FLOAT_EQ(pos.y(), static_cast<float>(i * 2.0f));
    EXPECT_FLOAT_EQ(pos.z(), static_cast<float>(i * 3.0f));
  }

  for (std::size_t i = 0; i < genericBlocksConstView.pca().metadata().size(); ++i) {
    auto pca_val = genericBlocksConstView.pca()[i];
    EXPECT_FLOAT_EQ(pca_val.vector_1(), static_cast<float>(i) / step);
    EXPECT_FLOAT_EQ(pca_val.vector_2(), static_cast<float>(i * 2.0f) / step);
    EXPECT_FLOAT_EQ(pca_val.vector_3(), static_cast<float>(i * 3.0f) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(0), static_cast<double>(i) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(1), static_cast<double>(i * 2.0) / step);
    EXPECT_DOUBLE_EQ(pca_val.candidateDirection()(2), static_cast<double>(i * 3.0) / step);
  }

  // Check mutation via original mutable view reflects in ConstView
  (*positionView)[3].x() = 0.0f;
  EXPECT_FLOAT_EQ(genericBlocksConstView.position()[3].x(), (*positionView)[3].x());
}

TEST_F(SoAGenericBlocksTest, DeepCopyGenericBlocksView) {
  SoAGenericBlocksView genericBlocksView{*positionView, *pcaView};

  std::array<ffx::soa::size_type, 2> sizes{{elemsPos, elemsPCA}};
  const std::size_t blocksBufferSize = SoAGenericBlocks::computeDataSize(sizes);
  std::unique_ptr<std::byte, decltype(&std::free)> bufferBlocks{
      reinterpret_cast<std::byte*>(aligned_alloc(SoAGenericBlocks::alignment, blocksBufferSize)), std::free};

  SoAGenericBlocks genericBlocks{bufferBlocks.get(), sizes};
  genericBlocks.deepCopy(genericBlocksView);

  SoAGenericBlocksView genericSoABlocksView{genericBlocks};

  // Check inequality of memory addresses (standalone buffer allocated)
  EXPECT_NE(genericSoABlocksView.position().metadata().addressOf_x(), positionConstView->metadata().addressOf_x());
  EXPECT_NE(genericSoABlocksView.position().metadata().addressOf_y(), positionConstView->metadata().addressOf_y());
  EXPECT_NE(genericSoABlocksView.position().metadata().addressOf_z(), positionConstView->metadata().addressOf_z());
  EXPECT_NE(genericSoABlocksView.pca().metadata().addressOf_candidateDirection(),
            pcaConstView->metadata().addressOf_candidateDirection());

  // Check contiguity of columns across block boundaries
  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoABlocksView.position().metadata().addressOf_x()) +
                ffx::soa::alignSize(elemsPos * sizeof(float), SoAGenericBlocks::alignment),
            reinterpret_cast<std::byte*>(genericSoABlocksView.position().metadata().addressOf_y()));

  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoABlocksView.position().metadata().addressOf_y()) +
                ffx::soa::alignSize(elemsPos * sizeof(float), SoAGenericBlocks::alignment),
            reinterpret_cast<std::byte*>(genericSoABlocksView.position().metadata().addressOf_z()));

  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoABlocksView.position().metadata().addressOf_z()) +
                ffx::soa::alignSize(elemsPos * sizeof(float), SoAGenericBlocks::alignment),
            reinterpret_cast<std::byte*>(genericSoABlocksView.position().metadata().addressOf_detectorType()));

  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoABlocksView.position().metadata().addressOf_detectorType()) +
                ffx::soa::alignSize(sizeof(int), SoAGenericBlocks::alignment),
            reinterpret_cast<std::byte*>(genericSoABlocksView.pca().metadata().addressOf_vector_1()));

  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoABlocksView.pca().metadata().addressOf_vector_1()) +
                ffx::soa::alignSize(elemsPCA * sizeof(float), SoAGenericBlocks::alignment),
            reinterpret_cast<std::byte*>(genericSoABlocksView.pca().metadata().addressOf_vector_2()));

  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoABlocksView.pca().metadata().addressOf_vector_2()) +
                ffx::soa::alignSize(elemsPCA * sizeof(float), SoAGenericBlocks::alignment),
            reinterpret_cast<std::byte*>(genericSoABlocksView.pca().metadata().addressOf_vector_3()));

  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoABlocksView.pca().metadata().addressOf_vector_3()) +
                ffx::soa::alignSize(elemsPCA * sizeof(float), SoAGenericBlocks::alignment),
            reinterpret_cast<std::byte*>(genericSoABlocksView.pca().metadata().addressOf_candidateDirection()));

  // Check values correctness
  for (std::size_t i = 0; i < elemsPos; ++i) {
    EXPECT_FLOAT_EQ(genericSoABlocksView.position()[i].x(), (*positionConstView)[i].x());
    EXPECT_FLOAT_EQ(genericSoABlocksView.position()[i].y(), (*positionConstView)[i].y());
    EXPECT_FLOAT_EQ(genericSoABlocksView.position()[i].z(), (*positionConstView)[i].z());
  }
  EXPECT_EQ(genericSoABlocksView.position().detectorType(), positionConstView->detectorType());

  for (std::size_t i = 0; i < elemsPCA; ++i) {
    EXPECT_FLOAT_EQ(genericSoABlocksView.pca()[i].vector_1(), (*pcaConstView)[i].vector_1());
    EXPECT_FLOAT_EQ(genericSoABlocksView.pca()[i].vector_2(), (*pcaConstView)[i].vector_2());
    EXPECT_FLOAT_EQ(genericSoABlocksView.pca()[i].vector_3(), (*pcaConstView)[i].vector_3());
    EXPECT_DOUBLE_EQ(genericSoABlocksView.pca()[i].candidateDirection()(0), (*pcaConstView)[i].candidateDirection()(0));
    EXPECT_DOUBLE_EQ(genericSoABlocksView.pca()[i].candidateDirection()(1), (*pcaConstView)[i].candidateDirection()(1));
    EXPECT_DOUBLE_EQ(genericSoABlocksView.pca()[i].candidateDirection()(2), (*pcaConstView)[i].candidateDirection()(2));
  }

  // Check independence of copied memory
  genericSoABlocksView.position()[3].x() = 0.0f;
  EXPECT_NE(genericSoABlocksView.position()[3].x(), (*positionView)[3].x());
}