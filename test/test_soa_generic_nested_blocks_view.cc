#include <cstdlib>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <array>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "ffx/ffx.h"

// Define SoA Layouts
GENERATE_SOA_LAYOUT(SoALayout1, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))
GENERATE_SOA_LAYOUT(SoALayout2, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))
GENERATE_SOA_LAYOUT(SoALayout3, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))

GENERATE_SOA_BLOCKS(BlocksTemplate, SOA_BLOCK(first, SoALayout1), SOA_BLOCK(second, SoALayout2))
GENERATE_SOA_BLOCKS(NestedBlocksTemplate, SOA_BLOCK(blocks, BlocksTemplate), SOA_BLOCK(soa, SoALayout3))

using BlocksSoA = BlocksTemplate<>;
using BlocksView = BlocksSoA::View;
using BlocksConstView = BlocksSoA::ConstView;

using NestedBlocksSoA = NestedBlocksTemplate<>;
using NestedBlocksView = NestedBlocksSoA::View;
using NestedBlocksConstView = NestedBlocksSoA::ConstView;

// Test Fixture for Nested SoA Blocks Tests
class SoAGenericNestedBlocksViewTest : public ::testing::Test {
protected:
  static constexpr std::array<NestedBlocksSoA::size_type, 3> sizes = {10, 20, 30};

  void SetUp() override {
    const auto bufferSize = NestedBlocksSoA::computeDataSize(sizes);
    buffer.reset(reinterpret_cast<std::byte*>(aligned_alloc(NestedBlocksSoA::alignment, bufferSize)));
    ASSERT_NE(buffer, nullptr);

    nestedBlocks = std::make_unique<NestedBlocksSoA>(buffer.get(), sizes);
    nestedBlocksView = std::make_unique<NestedBlocksView>(*nestedBlocks);
    nestedBlocksConstView = std::make_unique<NestedBlocksConstView>(*nestedBlocks);

    // Populate data
    nestedBlocksView->blocks().first().id() = 21;
    for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
      (*nestedBlocksView).blocks().first()[i].column() = static_cast<int>(i);
      (*nestedBlocksView).blocks().first()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    nestedBlocksView->blocks().second().id() = 42;
    for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
      (*nestedBlocksView).blocks().second()[i].column() = static_cast<int>(i);
      (*nestedBlocksView).blocks().second()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    nestedBlocksView->soa().id() = 666;
    for (NestedBlocksSoA::size_type i = 0; i < sizes[2]; ++i) {
      (*nestedBlocksView).soa()[i].column() = static_cast<int>(i);
      (*nestedBlocksView).soa()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }
  }

  std::unique_ptr<std::byte, decltype(&std::free)> buffer{nullptr, std::free};
  std::unique_ptr<NestedBlocksSoA> nestedBlocks;
  std::unique_ptr<NestedBlocksView> nestedBlocksView;
  std::unique_ptr<NestedBlocksConstView> nestedBlocksConstView;
};

TEST_F(SoAGenericNestedBlocksViewTest, ViewFromNestedBlocks) {
  BlocksView blocksView{nestedBlocksView->blocks().first(), nestedBlocksView->blocks().second()};

  // Verify metadata
  EXPECT_EQ(blocksView.metadata().size()[0], sizes[0]);
  EXPECT_EQ(blocksView.first().metadata().size(), sizes[0]);
  EXPECT_EQ(blocksView.metadata().size()[1], sizes[1]);
  EXPECT_EQ(blocksView.second().metadata().size(), sizes[1]);

  // Check equality of memory addresses
  EXPECT_EQ(blocksView.first().metadata().addressOf_column(),
            nestedBlocksView->blocks().first().metadata().addressOf_column());
  EXPECT_EQ(blocksView.first().metadata().addressOf_vector(),
            nestedBlocksView->blocks().first().metadata().addressOf_vector());
  EXPECT_EQ(blocksView.second().metadata().addressOf_column(),
            nestedBlocksView->blocks().second().metadata().addressOf_column());
  EXPECT_EQ(blocksView.second().metadata().addressOf_vector(),
            nestedBlocksView->blocks().second().metadata().addressOf_vector());

  // Verify data
  for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
    auto nestedFirst = nestedBlocksView->blocks().first()[i];
    auto first = blocksView.first()[i];
    EXPECT_EQ(first.column(), nestedFirst.column());
    EXPECT_EQ(first.vector(), nestedFirst.vector());
  }

  for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
    auto nestedSecond = nestedBlocksView->blocks().second()[i];
    auto second = blocksView.second()[i];
    EXPECT_EQ(second.column(), nestedSecond.column());
    EXPECT_EQ(second.vector(), nestedSecond.vector());
  }

  EXPECT_EQ(nestedBlocksView->blocks().first().id(), blocksView.first().id());
  EXPECT_EQ(nestedBlocksView->blocks().second().id(), blocksView.second().id());
}

TEST_F(SoAGenericNestedBlocksViewTest, ConstViewFromConstNestedBlocks) {
  BlocksConstView blocksConstView{nestedBlocksConstView->blocks().first(), nestedBlocksConstView->blocks().second()};

  // Verify metadata
  EXPECT_EQ(blocksConstView.metadata().size()[0], sizes[0]);
  EXPECT_EQ(blocksConstView.first().metadata().size(), sizes[0]);
  EXPECT_EQ(blocksConstView.metadata().size()[1], sizes[1]);
  EXPECT_EQ(blocksConstView.second().metadata().size(), sizes[1]);

  // Check equality of memory addresses
  EXPECT_EQ(blocksConstView.first().metadata().addressOf_column(),
            nestedBlocksConstView->blocks().first().metadata().addressOf_column());
  EXPECT_EQ(blocksConstView.first().metadata().addressOf_vector(),
            nestedBlocksConstView->blocks().first().metadata().addressOf_vector());
  EXPECT_EQ(blocksConstView.second().metadata().addressOf_column(),
            nestedBlocksConstView->blocks().second().metadata().addressOf_column());
  EXPECT_EQ(blocksConstView.second().metadata().addressOf_vector(),
            nestedBlocksConstView->blocks().second().metadata().addressOf_vector());

  // Verify data
  for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
    auto nestedFirst = nestedBlocksConstView->blocks().first()[i];
    auto first = blocksConstView.first()[i];
    EXPECT_EQ(first.column(), nestedFirst.column());
    EXPECT_EQ(first.vector(), nestedFirst.vector());
  }

  for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
    auto nestedSecond = nestedBlocksConstView->blocks().second()[i];
    auto second = blocksConstView.second()[i];
    EXPECT_EQ(second.column(), nestedSecond.column());
    EXPECT_EQ(second.vector(), nestedSecond.vector());
  }

  EXPECT_EQ(nestedBlocksConstView->blocks().first().id(), blocksConstView.first().id());
  EXPECT_EQ(nestedBlocksConstView->blocks().second().id(), blocksConstView.second().id());
}

TEST_F(SoAGenericNestedBlocksViewTest, ConstViewFromMutableNestedBlocks) {
  BlocksConstView blocksConstView{nestedBlocksView->blocks().first(), nestedBlocksView->blocks().second()};

  // Verify metadata
  EXPECT_EQ(blocksConstView.metadata().size()[0], sizes[0]);
  EXPECT_EQ(blocksConstView.first().metadata().size(), sizes[0]);
  EXPECT_EQ(blocksConstView.metadata().size()[1], sizes[1]);
  EXPECT_EQ(blocksConstView.second().metadata().size(), sizes[1]);

  // Check equality of memory addresses
  EXPECT_EQ(blocksConstView.first().metadata().addressOf_column(),
            nestedBlocksConstView->blocks().first().metadata().addressOf_column());
  EXPECT_EQ(blocksConstView.first().metadata().addressOf_vector(),
            nestedBlocksConstView->blocks().first().metadata().addressOf_vector());
  EXPECT_EQ(blocksConstView.second().metadata().addressOf_column(),
            nestedBlocksConstView->blocks().second().metadata().addressOf_column());
  EXPECT_EQ(blocksConstView.second().metadata().addressOf_vector(),
            nestedBlocksConstView->blocks().second().metadata().addressOf_vector());

  // Verify data
  for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
    auto nestedFirst = nestedBlocksConstView->blocks().first()[i];
    auto first = blocksConstView.first()[i];
    EXPECT_EQ(first.column(), nestedFirst.column());
    EXPECT_EQ(first.vector(), nestedFirst.vector());
  }

  for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
    auto nestedSecond = nestedBlocksConstView->blocks().second()[i];
    auto second = blocksConstView.second()[i];
    EXPECT_EQ(second.column(), nestedSecond.column());
    EXPECT_EQ(second.vector(), nestedSecond.vector());
  }

  EXPECT_EQ(nestedBlocksConstView->blocks().first().id(), blocksConstView.first().id());
  EXPECT_EQ(nestedBlocksConstView->blocks().second().id(), blocksConstView.second().id());
}

TEST_F(SoAGenericNestedBlocksViewTest, DeepCopyNestedBlocksToNormalBlocksLayout) {
  BlocksView genericBlocksView{nestedBlocksView->blocks().first(), nestedBlocksView->blocks().second()};

  std::array<NestedBlocksSoA::size_type, 2> size = {sizes[0], sizes[1]};
  const std::size_t blocksBufferSize = BlocksSoA::computeDataSize(size);
  std::unique_ptr<std::byte, decltype(&std::free)> bufferBlocks{
      reinterpret_cast<std::byte*>(aligned_alloc(BlocksSoA::alignment, blocksBufferSize)), std::free};

  BlocksSoA genericBlocks{bufferBlocks.get(), size};
  genericBlocks.deepCopy(genericBlocksView);

  BlocksView genericSoABlocksView{genericBlocks};

  // Check inequality of memory addresses (standalone buffer allocated)
  EXPECT_NE(genericSoABlocksView.first().metadata().addressOf_column(),
            nestedBlocksView->blocks().first().metadata().addressOf_column());
  EXPECT_NE(genericSoABlocksView.first().metadata().addressOf_vector(),
            nestedBlocksView->blocks().first().metadata().addressOf_vector());
  EXPECT_NE(genericSoABlocksView.second().metadata().addressOf_column(),
            nestedBlocksView->blocks().second().metadata().addressOf_column());
  EXPECT_NE(genericSoABlocksView.second().metadata().addressOf_vector(),
            nestedBlocksView->blocks().second().metadata().addressOf_vector());

  // Verify data
  for (NestedBlocksSoA::size_type i = 0; i < sizes[0]; ++i) {
    auto nestedFirst = nestedBlocksConstView->blocks().first()[i];
    auto first = genericSoABlocksView.first()[i];
    EXPECT_EQ(first.column(), nestedFirst.column());
    EXPECT_EQ(first.vector(), nestedFirst.vector());
  }

  for (NestedBlocksSoA::size_type i = 0; i < sizes[1]; ++i) {
    auto nestedSecond = nestedBlocksConstView->blocks().second()[i];
    auto second = genericSoABlocksView.second()[i];
    EXPECT_EQ(second.column(), nestedSecond.column());
    EXPECT_EQ(second.vector(), nestedSecond.vector());
  }

  EXPECT_EQ(nestedBlocksConstView->blocks().first().id(), genericSoABlocksView.first().id());
  EXPECT_EQ(nestedBlocksConstView->blocks().second().id(), genericSoABlocksView.second().id());
}