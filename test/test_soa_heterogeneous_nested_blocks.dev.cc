#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <alpaka/alpaka.hpp>

#include "ffx/ffx.h"

using namespace ffx_runtime;

GENERATE_SOA_LAYOUT(SoALayout1, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))
GENERATE_SOA_LAYOUT(SoALayout2, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))
GENERATE_SOA_LAYOUT(SoALayout3, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))
GENERATE_SOA_LAYOUT(SoALayout4, SOA_COLUMN(int, column), SOA_EIGEN_COLUMN(Eigen::Vector3d, vector), SOA_SCALAR(int, id))

GENERATE_SOA_BLOCKS(BlocksTemplate, SOA_BLOCK(first, SoALayout1), SOA_BLOCK(second, SoALayout2))

GENERATE_SOA_BLOCKS(SingleNestedBlocksTemplate, SOA_BLOCK(blocks, BlocksTemplate), SOA_BLOCK(soa, SoALayout3))

GENERATE_SOA_BLOCKS(DoubleNestedBlocksTemplate,
                    SOA_BLOCK(blocks, SingleNestedBlocksTemplate),
                    SOA_BLOCK(soa, SoALayout4))

using BlockSoA = DoubleNestedBlocksTemplate<>;
using View = BlockSoA::View;
using ConstView = BlockSoA::ConstView;

// Fill SoAs Kernel
struct FillSoAs {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, View view) const {
    // Fill elements of SoALayout1
    if (alpaka::oncePerGrid(acc)) {
      view.blocks().blocks().first().id() = static_cast<int>(view.metadata().size()[0]);
    }
    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[0])) {
      auto element = view.blocks().blocks().first()[i];
      element.column() = static_cast<int>(i);
      element.vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    // Fill elements of SoALayout2
    if (alpaka::oncePerGrid(acc)) {
      view.blocks().blocks().second().id() = static_cast<int>(view.metadata().size()[1]);
    }
    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[1])) {
      auto element = view.blocks().blocks().second()[i];
      element.column() = static_cast<int>(i * 10);
      element.vector() = Eigen::Vector3d(i * 10, i * 10 + 1, i * 10 + 2);
    }

    // Fill elements of SoALayout3
    if (alpaka::oncePerGrid(acc)) {
      view.blocks().soa().id() = static_cast<int>(view.metadata().size()[2]);
    }
    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[2])) {
      auto element = view.blocks().soa()[i];
      element.column() = static_cast<int>(i * 100);
      element.vector() = Eigen::Vector3d(i * 100, i * 100 + 1, i * 100 + 2);
    }

    // Fill elements of SoALayout4
    if (alpaka::oncePerGrid(acc)) {
      view.soa().id() = static_cast<int>(view.metadata().size()[3]);
    }
    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[3])) {
      auto element = view.soa()[i];
      element.column() = static_cast<int>(i * 6654);
      element.vector() = Eigen::Vector3d(i * 6654, i * 6654 + 1, i * 6654 + 2);
    }
  }
};

namespace {

  void checkNestedSoABlocks(const ConstView& view) {
    EXPECT_EQ(view.blocks().blocks().first().id(), static_cast<int>(view.metadata().size()[0]));
    for (BlockSoA::size_type i = 0; i < view.metadata().size()[0]; ++i) {
      const auto& element = view.blocks().blocks().first()[i];
      EXPECT_EQ(element.column(), static_cast<int>(i));
      EXPECT_TRUE(element.vector().isApprox(Eigen::Vector3d(i, i + 1, i + 2)));
    }

    EXPECT_EQ(view.blocks().blocks().second().id(), static_cast<int>(view.metadata().size()[1]));
    for (BlockSoA::size_type i = 0; i < view.metadata().size()[1]; ++i) {
      const auto& element = view.blocks().blocks().second()[i];
      EXPECT_EQ(element.column(), static_cast<int>(i * 10));
      EXPECT_TRUE(element.vector().isApprox(Eigen::Vector3d(i * 10, i * 10 + 1, i * 10 + 2)));
    }

    EXPECT_EQ(view.blocks().soa().id(), static_cast<int>(view.metadata().size()[2]));
    for (BlockSoA::size_type i = 0; i < view.metadata().size()[2]; ++i) {
      const auto& element = view.blocks().soa()[i];
      EXPECT_EQ(element.column(), static_cast<int>(i * 100));
      EXPECT_TRUE(element.vector().isApprox(Eigen::Vector3d(i * 100, i * 100 + 1, i * 100 + 2)));
    }

    EXPECT_EQ(view.soa().id(), static_cast<int>(view.metadata().size()[3]));
    for (BlockSoA::size_type i = 0; i < view.metadata().size()[3]; ++i) {
      const auto& element = view.soa()[i];
      EXPECT_EQ(element.column(), static_cast<int>(i * 6654));
      EXPECT_TRUE(element.vector().isApprox(Eigen::Vector3d(i * 6654, i * 6654 + 1, i * 6654 + 2)));
    }
  }

}  // namespace

class SoAPortableDoubleNestedBlocksAlpakaTest : public ::testing::Test {
protected:
  static constexpr std::array<BlockSoA::size_type, 4> sizes = {2, 1189, 33, 3333};
};

TEST_F(SoAPortableDoubleNestedBlocksAlpakaTest, NestedSoABlocksPropagationAndExecution) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty()) {
    GTEST_SKIP() << "No devices available for the " << BOOST_PP_STRINGIZE(ffx_runtime) << " backend, skipping.";
  }

  for (auto const& device : devices) {
    Queue queue(device);

    const std::size_t N = *std::max_element(sizes.begin(), sizes.end());

    ffx::soa::PortableCollection<Device, BlockSoA> nestedBlocksCollection(queue, sizes);
    View view = nestedBlocksCollection.view();

    // Check size propagation across view levels
    EXPECT_EQ(view.metadata().size()[0], sizes[0]);
    EXPECT_EQ(view.blocks().blocks().first().metadata().size(), sizes[0]);
    EXPECT_EQ(view.metadata().size()[1], sizes[1]);
    EXPECT_EQ(view.blocks().blocks().second().metadata().size(), sizes[1]);
    EXPECT_EQ(view.metadata().size()[2], sizes[2]);
    EXPECT_EQ(view.blocks().soa().metadata().size(), sizes[2]);
    EXPECT_EQ(view.metadata().size()[3], sizes[3]);
    EXPECT_EQ(view.soa().metadata().size(), sizes[3]);

    // Work division setup
    constexpr std::size_t blockSize = 256;
    const std::size_t nBlocks = ffx::divide_up_by(N, blockSize);
    const auto workDiv = ffx::make_workdiv<Acc1D>(nBlocks, blockSize);

    // Execute Fill Kernel
    alpaka::exec<Acc1D>(queue, workDiv, FillSoAs{}, view);
    alpaka::wait(queue);

    // Check results on host
    ffx::soa::PortableHostCollection<BlockSoA> nestedBlocksHostCollection(ffx::host(), sizes);
    alpaka::memcpy(queue, nestedBlocksHostCollection.buffer(), nestedBlocksCollection.buffer());
    alpaka::wait(queue);

    ConstView constHostView = nestedBlocksHostCollection.const_view();
    checkNestedSoABlocks(constHostView);
  }
}