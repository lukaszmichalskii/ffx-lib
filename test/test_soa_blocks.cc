#include <cstdlib>
#include <memory>
#include <array>
#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "ffx/ffx.h"

// Define test layouts
GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, vector_1),
                    SOA_COLUMN(float, vector_2),
                    SOA_COLUMN(float, vector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

GENERATE_SOA_LAYOUT(SoATemplate, SOA_SCALAR(int, id), SOA_SCALAR(int, type), SOA_SCALAR(float, energy))

GENERATE_SOA_LAYOUT(
    SimpleLayoutTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z), SOA_COLUMN(float, t))

// Define test block containers
GENERATE_SOA_BLOCKS(SoABlocksTemplate,
                    SOA_BLOCK(position, SoAPositionTemplate),
                    SOA_BLOCK(pca, SoAPCATemplate),
                    SOA_BLOCK(scalars, SoATemplate))

GENERATE_SOA_BLOCKS(NestedBlocksTemplate, SOA_BLOCK(blocks, SoABlocksTemplate), SOA_BLOCK(simple, SimpleLayoutTemplate))

using SoABlocks = SoABlocksTemplate<>;
using SoABlocksView = SoABlocks::View;
using SoABlocksConstView = SoABlocks::ConstView;

using NestedBlocks = NestedBlocksTemplate<>;
using NestedBlocksView = NestedBlocks::View;
using NestedBlocksConstView = NestedBlocks::ConstView;

// Test Fixture for sharing allocated memory and initialized data
class SoABlocksTest : public ::testing::Test {
protected:
  void SetUp() override {
    sizes = {{10, 20, 1}};
    blocksBufferSize = SoABlocks::computeDataSize(sizes);

    buffer.reset(reinterpret_cast<std::byte*>(aligned_alloc(SoABlocks::alignment, blocksBufferSize)));

    blocks = std::make_unique<SoABlocks>(buffer.get(), sizes);
    blocksView = std::make_unique<SoABlocksView>(*blocks);
    blocksConstView = std::make_unique<SoABlocksConstView>(*blocks);

    // Populate default data
    blocksView->position().detectorType() = 1;
    for (std::size_t i = 0; i < blocksView->position().metadata().size(); ++i) {
      (*blocksView).position()[i] = {0.1f, 0.2f, 0.3f};
    }
    for (std::size_t i = 0; i < blocksView->metadata().size()[1]; ++i) {
      (*blocksView).pca()[i].vector_1() = 0.0f;
      (*blocksView).pca()[i].vector_2() = 0.0f;
      (*blocksView).pca()[i].vector_3() = 1.0f;
      (*blocksView).pca()[i].candidateDirection() = Eigen::Vector3d(1.0, 0.0, 0.0);
    }
    blocksView->scalars().id() = 42;
    blocksView->scalars().type() = 1;
    blocksView->scalars().energy() = 100.0f;
  }

  std::array<ffx::soa::size_type, 3> sizes;
  std::size_t blocksBufferSize{0};
  std::unique_ptr<std::byte, decltype(&std::free)> buffer{nullptr, std::free};

  std::unique_ptr<SoABlocks> blocks;
  std::unique_ptr<SoABlocksView> blocksView;
  std::unique_ptr<SoABlocksConstView> blocksConstView;
};

TEST_F(SoABlocksTest, AlignmentAndEnforcementDefaults) {
  EXPECT_EQ(SoABlocks::alignment, ffx::soa::CacheLineSize::defaultSize);
  EXPECT_EQ(SoABlocks::alignmentEnforcement, ffx::soa::AlignmentEnforcement::relaxed);

  EXPECT_EQ(blocks->position().alignment, ffx::soa::CacheLineSize::defaultSize);
  EXPECT_EQ(blocks->position().alignmentEnforcement, ffx::soa::AlignmentEnforcement::relaxed);

  EXPECT_EQ(blocks->pca().alignment, ffx::soa::CacheLineSize::defaultSize);
  EXPECT_EQ(blocks->pca().alignmentEnforcement, ffx::soa::AlignmentEnforcement::relaxed);

  EXPECT_EQ(blocks->scalars().alignment, ffx::soa::CacheLineSize::defaultSize);
  EXPECT_EQ(blocks->scalars().alignmentEnforcement, ffx::soa::AlignmentEnforcement::relaxed);

  // Verify memory layout adjacency
  EXPECT_EQ(blocks->position().metadata().nextByte(), blocks->metadata().addressOf_pca());
  EXPECT_EQ(blocks->pca().metadata().nextByte(), blocks->metadata().addressOf_scalars());
}

TEST_F(SoABlocksTest, ViewAccessAndMetadata) {
  EXPECT_EQ(blocksView->metadata().size()[0], 10);
  EXPECT_EQ(blocksView->position().metadata().size(), 10);
  EXPECT_EQ(blocksView->metadata().size()[1], 20);
  EXPECT_EQ(blocksView->pca().metadata().size(), 20);
  EXPECT_EQ(blocksView->metadata().size()[2], 1);
  EXPECT_EQ(blocksView->scalars().metadata().size(), 1);

  for (std::size_t i = 0; i < blocksView->position().metadata().size(); ++i) {
    auto pos = (*blocksView).position()[i];
    EXPECT_FLOAT_EQ(pos.x(), 0.1f);
    EXPECT_FLOAT_EQ(pos.y(), 0.2f);
    EXPECT_FLOAT_EQ(pos.z(), 0.3f);
  }

  for (std::size_t i = 0; i < blocksView->pca().metadata().size(); ++i) {
    auto pca = (*blocksView).pca()[i];
    EXPECT_FLOAT_EQ(pca.vector_1(), 0.0f);
    EXPECT_FLOAT_EQ(pca.vector_2(), 0.0f);
    EXPECT_FLOAT_EQ(pca.vector_3(), 1.0f);
    EXPECT_DOUBLE_EQ(pca.candidateDirection()(0), 1.0);
    EXPECT_DOUBLE_EQ(pca.candidateDirection()(1), 0.0);
    EXPECT_DOUBLE_EQ(pca.candidateDirection()(2), 0.0);
  }
}

TEST_F(SoABlocksTest, ConstViewAccessAndMetadata) {
  EXPECT_EQ(blocksConstView->metadata().size()[0], 10);
  EXPECT_EQ(blocksConstView->position().metadata().size(), 10);
  EXPECT_EQ(blocksConstView->metadata().size()[1], 20);
  EXPECT_EQ(blocksConstView->pca().metadata().size(), 20);
  EXPECT_EQ(blocksConstView->metadata().size()[2], 1);
  EXPECT_EQ(blocksConstView->scalars().metadata().size(), 1);

  for (std::size_t i = 0; i < blocksConstView->position().metadata().size(); ++i) {
    auto pos = (*blocksConstView).position()[i];
    EXPECT_FLOAT_EQ(pos.x(), 0.1f);
    EXPECT_FLOAT_EQ(pos.y(), 0.2f);
    EXPECT_FLOAT_EQ(pos.z(), 0.3f);
  }

  for (std::size_t i = 0; i < blocksConstView->pca().metadata().size(); ++i) {
    auto pca = (*blocksConstView).pca()[i];
    EXPECT_FLOAT_EQ(pca.vector_1(), 0.0f);
    EXPECT_FLOAT_EQ(pca.vector_2(), 0.0f);
    EXPECT_FLOAT_EQ(pca.vector_3(), 1.0f);
    EXPECT_DOUBLE_EQ(pca.candidateDirection()(0), 1.0);
    EXPECT_DOUBLE_EQ(pca.candidateDirection()(1), 0.0);
    EXPECT_DOUBLE_EQ(pca.candidateDirection()(2), 0.0);
  }
}

TEST_F(SoABlocksTest, RangeCheckingView) {
  int underflow = -1;
  int overflow = static_cast<int>(blocksView->position().metadata().size());

  // Check row accessor boundaries
  EXPECT_THROW((*blocksView).position()[underflow], std::out_of_range);
  EXPECT_THROW((*blocksView).position()[overflow], std::out_of_range);

  // Check column element accessor boundaries
  EXPECT_THROW(blocksView->position().x(underflow), std::out_of_range);
  EXPECT_THROW(blocksView->position().x(overflow), std::out_of_range);
}

TEST_F(SoABlocksTest, RangeCheckingConstView) {
  int underflow = -1;
  int overflow = static_cast<int>(blocksConstView->pca().metadata().size());

  // Check row accessor boundaries
  EXPECT_THROW((*blocksConstView).pca()[underflow], std::out_of_range);
  EXPECT_THROW((*blocksConstView).pca()[overflow], std::out_of_range);

  // Check column element accessor boundaries
  EXPECT_THROW(blocksConstView->pca().vector_1(underflow), std::out_of_range);
  EXPECT_THROW(blocksConstView->pca().vector_1(overflow), std::out_of_range);
}

TEST_F(SoABlocksTest, CustomAlignmentAndEnforcementTemplates) {
  static constexpr ffx::soa::byte_size_type testAlignment = 256;
  static constexpr bool alignmentEnforcement = ffx::soa::AlignmentEnforcement::enforced;

  using SoABlocksTemplated = SoABlocksTemplate<testAlignment, alignmentEnforcement>;

  std::array<ffx::soa::size_type, 3> localSizes{{10, 20, 1}};
  const std::size_t templatedBufferSize = SoABlocksTemplated::computeDataSize(localSizes);

  std::unique_ptr<std::byte, decltype(&std::free)> localBuffer{
      reinterpret_cast<std::byte*>(aligned_alloc(SoABlocksTemplated::alignment, templatedBufferSize)), std::free};

  SoABlocksTemplated blocksTemplated(localBuffer.get(), localSizes);

  EXPECT_EQ(SoABlocksTemplated::alignment, testAlignment);
  EXPECT_EQ(SoABlocksTemplated::alignmentEnforcement, alignmentEnforcement);

  EXPECT_EQ(blocksTemplated.position().alignment, testAlignment);
  EXPECT_EQ(blocksTemplated.position().alignmentEnforcement, alignmentEnforcement);

  EXPECT_EQ(blocksTemplated.pca().alignment, testAlignment);
  EXPECT_EQ(blocksTemplated.pca().alignmentEnforcement, alignmentEnforcement);

  EXPECT_EQ(blocksTemplated.scalars().alignment, testAlignment);
  EXPECT_EQ(blocksTemplated.scalars().alignmentEnforcement, alignmentEnforcement);
}

TEST_F(SoABlocksTest, ViewTemplateParameters) {
  using NoRangeCheckBlockView =
      SoABlocks::ViewTemplate<ffx::soa::restrict_qualify::default_value, ffx::soa::range_checking::disabled>;
  NoRangeCheckBlockView noRangeCheckBlockView{*blocks};

  EXPECT_EQ(noRangeCheckBlockView.restrictQualify, ffx::soa::restrict_qualify::default_value);
  EXPECT_EQ(noRangeCheckBlockView.rangeChecking, ffx::soa::range_checking::disabled);
  EXPECT_EQ(noRangeCheckBlockView.position().restrictQualify, ffx::soa::restrict_qualify::default_value);
  EXPECT_EQ(noRangeCheckBlockView.position().rangeChecking, ffx::soa::range_checking::disabled);

  using NoRestrictBlockView =
      SoABlocks::ViewTemplate<ffx::soa::restrict_qualify::disabled, ffx::soa::range_checking::default_value>;
  NoRestrictBlockView noRestrictBlockView{*blocks};

  EXPECT_EQ(noRestrictBlockView.restrictQualify, ffx::soa::restrict_qualify::disabled);
  EXPECT_EQ(noRestrictBlockView.rangeChecking, ffx::soa::range_checking::default_value);
  EXPECT_EQ(noRestrictBlockView.position().restrictQualify, ffx::soa::restrict_qualify::disabled);
  EXPECT_EQ(noRestrictBlockView.position().rangeChecking, ffx::soa::range_checking::default_value);
}

TEST(NestedBlocksTest, ExtendedBlocksLayout) {
  std::array<ffx::soa::size_type, 4> sizes{{11, 12, 13, 14}};
  const std::size_t blocksExtendedBufferSize = NestedBlocks::computeDataSize(sizes);

  std::unique_ptr<std::byte, decltype(&std::free)> buffer{
      reinterpret_cast<std::byte*>(aligned_alloc(NestedBlocks::alignment, blocksExtendedBufferSize)), std::free};

  NestedBlocks nestedBlocksSoA(buffer.get(), sizes);
  NestedBlocksView nestedBlocksView{nestedBlocksSoA};
  NestedBlocksConstView nestedBlocksConstView{nestedBlocksSoA};

  nestedBlocksView.blocks().position().detectorType() = 1;
  for (std::size_t i = 0; i < nestedBlocksView.metadata().size()[0]; ++i) {
    nestedBlocksView.blocks().position()[i] = {0.1f, 0.2f, 0.3f};
  }

  for (std::size_t i = 0; i < nestedBlocksView.metadata().size()[1]; ++i) {
    nestedBlocksView.blocks().pca()[i].vector_1() = 0.0f;
    nestedBlocksView.blocks().pca()[i].vector_2() = 0.0f;
    nestedBlocksView.blocks().pca()[i].vector_3() = 1.0f;
    nestedBlocksView.blocks().pca()[i].candidateDirection() = Eigen::Vector3d(1.0, 0.0, 0.0);
  }
  nestedBlocksView.blocks().scalars().id() = 42;
  nestedBlocksView.blocks().scalars().type() = 1;
  nestedBlocksView.blocks().scalars().energy() = 100.0f;

  for (std::size_t i = 0; i < nestedBlocksView.metadata().size()[3]; ++i) {
    nestedBlocksView.simple()[i] = {2.1f, 2.2f, 2.3f, 2.4f};
  }

  EXPECT_EQ(nestedBlocksSoA.blocks().position().metadata().size(), 11);
  EXPECT_EQ(nestedBlocksSoA.blocks().pca().metadata().size(), 12);
  EXPECT_EQ(nestedBlocksSoA.blocks().scalars().metadata().size(), 13);
  EXPECT_EQ(nestedBlocksSoA.simple().metadata().size(), 14);

  EXPECT_EQ(nestedBlocksConstView.blocks().position().detectorType(), 1);
  for (std::size_t i = 0; i < nestedBlocksConstView.metadata().size()[0]; ++i) {
    EXPECT_FLOAT_EQ(nestedBlocksConstView.blocks().position()[i].x(), 0.1f);
    EXPECT_FLOAT_EQ(nestedBlocksConstView.blocks().position()[i].y(), 0.2f);
    EXPECT_FLOAT_EQ(nestedBlocksConstView.blocks().position()[i].z(), 0.3f);
  }

  for (std::size_t i = 0; i < nestedBlocksConstView.metadata().size()[1]; ++i) {
    EXPECT_FLOAT_EQ(nestedBlocksConstView.blocks().pca()[i].vector_1(), 0.0f);
    EXPECT_FLOAT_EQ(nestedBlocksConstView.blocks().pca()[i].vector_2(), 0.0f);
    EXPECT_FLOAT_EQ(nestedBlocksConstView.blocks().pca()[i].vector_3(), 1.0f);
    EXPECT_DOUBLE_EQ(nestedBlocksConstView.blocks().pca()[i].candidateDirection()[0], 1.0);
    EXPECT_DOUBLE_EQ(nestedBlocksConstView.blocks().pca()[i].candidateDirection()[1], 0.0);
    EXPECT_DOUBLE_EQ(nestedBlocksConstView.blocks().pca()[i].candidateDirection()[2], 0.0);
  }

  EXPECT_EQ(nestedBlocksConstView.blocks().scalars().id(), 42);
  EXPECT_EQ(nestedBlocksConstView.blocks().scalars().type(), 1);
  EXPECT_FLOAT_EQ(nestedBlocksConstView.blocks().scalars().energy(), 100.0f);

  for (std::size_t i = 0; i < nestedBlocksConstView.metadata().size()[3]; ++i) {
    EXPECT_FLOAT_EQ(nestedBlocksConstView.simple()[i].x(), 2.1f);
    EXPECT_FLOAT_EQ(nestedBlocksConstView.simple()[i].y(), 2.2f);
    EXPECT_FLOAT_EQ(nestedBlocksConstView.simple()[i].z(), 2.3f);
    EXPECT_FLOAT_EQ(nestedBlocksConstView.simple()[i].t(), 2.4f);
  }
}