#include "../include/ffx/core/alpaka/config.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

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

GENERATE_SOA_BLOCKS(FirstBlocksTemplate, SOA_BLOCK(first, SoALayout1), SOA_BLOCK(second, SoALayout2))
GENERATE_SOA_BLOCKS(SecondBlocksTemplate, SOA_BLOCK(first, SoALayout3), SOA_BLOCK(second, SoALayout4))

GENERATE_SOA_BLOCKS(NestedBlocksTemplate,
                    SOA_BLOCK(firstBlocks, FirstBlocksTemplate),
                    SOA_BLOCK(secondBlocks, SecondBlocksTemplate),
                    SOA_BLOCK(firstLayout, SoALayout1),
                    SOA_BLOCK(secondLayout, SoALayout4))

GENERATE_SOA_BLOCKS(GenericBlocksTemplate, SOA_BLOCK(blocks, FirstBlocksTemplate), SOA_BLOCK(layout, SoALayout4))

using NestedBlocksSoA = NestedBlocksTemplate<>;
using NestedBlocksView = NestedBlocksSoA::View;
using NestedBlocksConstView = NestedBlocksSoA::ConstView;

using GenericSoA = GenericBlocksTemplate<>;
using GenericSoAView = GenericSoA::View;
using GenericSoAConstView = GenericSoA::ConstView;

// Kernel to fill SoA structures
struct FillSoA {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, NestedBlocksView view) const {
    if (alpaka::oncePerGrid(acc)) {
      view.firstBlocks().first().id() = 21;
      view.firstBlocks().second().id() = 22;
      view.secondBlocks().first().id() = 42;
      view.secondBlocks().second().id() = 43;
      view.firstLayout().id() = 333;
      view.secondLayout().id() = 666;
    }

    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[0])) {
      view.firstBlocks().first()[i].column() = static_cast<int>(i);
      view.firstBlocks().first()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[1])) {
      view.firstBlocks().second()[i].column() = static_cast<int>(i);
      view.firstBlocks().second()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[2])) {
      view.secondBlocks().first()[i].column() = static_cast<int>(i);
      view.secondBlocks().first()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[3])) {
      view.secondBlocks().second()[i].column() = static_cast<int>(i);
      view.secondBlocks().second()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[4])) {
      view.firstLayout()[i].column() = static_cast<int>(i);
      view.firstLayout()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }

    for (auto i : alpaka::uniformElements(acc, view.metadata().size()[5])) {
      view.secondLayout()[i].column() = static_cast<int>(i);
      view.secondLayout()[i].vector() = Eigen::Vector3d(i, i + 1, i + 2);
    }
  }
};

namespace {

  void checkDataAndMetadata(NestedBlocksConstView nestedBlocksConstView, GenericSoAConstView genericSoABlocksView) {
    EXPECT_EQ(nestedBlocksConstView.metadata().size()[0], genericSoABlocksView.metadata().size()[0]);
    EXPECT_EQ(nestedBlocksConstView.metadata().size()[1], genericSoABlocksView.metadata().size()[1]);
    EXPECT_EQ(nestedBlocksConstView.metadata().size()[5], genericSoABlocksView.metadata().size()[2]);

    // Verify data
    for (NestedBlocksSoA::size_type i = 0; i < genericSoABlocksView.metadata().size()[0]; ++i) {
      auto nestedFirst = nestedBlocksConstView.firstBlocks().first()[i];
      auto first = genericSoABlocksView.blocks().first()[i];
      EXPECT_EQ(first.column(), nestedFirst.column());
      EXPECT_EQ(first.vector(), nestedFirst.vector());
    }

    for (NestedBlocksSoA::size_type i = 0; i < genericSoABlocksView.metadata().size()[1]; ++i) {
      auto nestedSecond = nestedBlocksConstView.firstBlocks().second()[i];
      auto second = genericSoABlocksView.blocks().second()[i];
      EXPECT_EQ(second.column(), nestedSecond.column());
      EXPECT_EQ(second.vector(), nestedSecond.vector());
    }

    for (NestedBlocksSoA::size_type i = 0; i < genericSoABlocksView.metadata().size()[2]; ++i) {
      auto nested = nestedBlocksConstView.secondLayout()[i];
      auto generic = genericSoABlocksView.layout()[i];
      EXPECT_EQ(generic.column(), nested.column());
      EXPECT_EQ(generic.vector(), nested.vector());
    }

    EXPECT_EQ(nestedBlocksConstView.firstBlocks().first().id(), genericSoABlocksView.blocks().first().id());
    EXPECT_EQ(nestedBlocksConstView.firstBlocks().second().id(), genericSoABlocksView.blocks().second().id());
    EXPECT_EQ(nestedBlocksConstView.secondLayout().id(), genericSoABlocksView.layout().id());
  }

}  // namespace

class SoAPortableNestedBlocksAlpakaTest : public ::testing::Test {
protected:
  static constexpr std::array<NestedBlocksSoA::size_type, 6> sizes = {21, 45, 137, 43, 222, 177};

  void SetUpNestedCollection(Queue& queue,
                             ffx::soa::PortableCollection<Device, NestedBlocksSoA>& nestedBlocksCollection,
                             ffx::soa::PortableHostCollection<NestedBlocksSoA>& h_nestedBlocksCollection) {
    NestedBlocksView nestedBlocksView = nestedBlocksCollection.view();

    constexpr auto blockSize = 64;
    NestedBlocksSoA::size_type largestSize = *std::max_element(sizes.begin(), sizes.end());
    auto numberOfBlocks = ffx::divide_up_by(largestSize, blockSize);
    const auto workDiv = ffx::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue, workDiv, FillSoA{}, nestedBlocksView);
    alpaka::wait(queue);

    alpaka::memcpy(queue, h_nestedBlocksCollection.buffer(), nestedBlocksCollection.buffer());
    alpaka::wait(queue);
  }
};

TEST_F(SoAPortableNestedBlocksAlpakaTest, DeepCopyViewHostToHostAndDeviceToDevice) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty()) {
    GTEST_SKIP() << "No devices available for the " << BOOST_PP_STRINGIZE(ffx_runtime) << " backend, skipping.";
  }

  for (auto const& device : devices) {
    Queue queue(device);

    ffx::soa::PortableCollection<Device, NestedBlocksSoA> nestedBlocksCollection(queue, sizes);
    ffx::soa::PortableHostCollection<NestedBlocksSoA> h_nestedBlocksCollection(queue, sizes);
    SetUpNestedCollection(queue, nestedBlocksCollection, h_nestedBlocksCollection);

    NestedBlocksView nestedBlocksView = nestedBlocksCollection.view();

    GenericSoAView genericView{nestedBlocksView.firstBlocks(), nestedBlocksView.secondLayout()};

    // Verify metadata
    EXPECT_EQ(genericView.metadata().size()[0], sizes[0]);
    EXPECT_EQ(genericView.blocks().first().metadata().size(), sizes[0]);
    EXPECT_EQ(genericView.metadata().size()[1], sizes[1]);
    EXPECT_EQ(genericView.blocks().second().metadata().size(), sizes[1]);
    EXPECT_EQ(genericView.metadata().size()[2], sizes[5]);
    EXPECT_EQ(genericView.layout().metadata().size(), sizes[5]);

    // Check equality of memory addresses
    EXPECT_EQ(genericView.blocks().first().metadata().addressOf_column(),
              nestedBlocksView.firstBlocks().first().metadata().addressOf_column());
    EXPECT_EQ(genericView.blocks().first().metadata().addressOf_vector(),
              nestedBlocksView.firstBlocks().first().metadata().addressOf_vector());
    EXPECT_EQ(genericView.blocks().second().metadata().addressOf_column(),
              nestedBlocksView.firstBlocks().second().metadata().addressOf_column());
    EXPECT_EQ(genericView.blocks().second().metadata().addressOf_vector(),
              nestedBlocksView.firstBlocks().second().metadata().addressOf_vector());
    EXPECT_EQ(genericView.layout().metadata().addressOf_column(),
              nestedBlocksView.secondLayout().metadata().addressOf_column());
    EXPECT_EQ(genericView.layout().metadata().addressOf_vector(),
              nestedBlocksView.secondLayout().metadata().addressOf_vector());

    // PortableCollection hosting aggregated columns
    ffx::soa::PortableCollection<Device, GenericSoA> blocksCollection(queue, sizes[0], sizes[1], sizes[5]);
    blocksCollection.deepCopy(queue, genericView);

    GenericSoAView copiedBlocksView = blocksCollection.view();
    EXPECT_NE(copiedBlocksView.blocks().first().metadata().addressOf_column(),
              nestedBlocksView.firstBlocks().first().metadata().addressOf_column());
    EXPECT_NE(copiedBlocksView.blocks().first().metadata().addressOf_vector(),
              nestedBlocksView.firstBlocks().first().metadata().addressOf_vector());
    EXPECT_NE(copiedBlocksView.blocks().second().metadata().addressOf_column(),
              nestedBlocksView.firstBlocks().second().metadata().addressOf_column());
    EXPECT_NE(copiedBlocksView.blocks().second().metadata().addressOf_vector(),
              nestedBlocksView.firstBlocks().second().metadata().addressOf_vector());
    EXPECT_NE(copiedBlocksView.layout().metadata().addressOf_column(),
              nestedBlocksView.secondLayout().metadata().addressOf_column());
    EXPECT_NE(copiedBlocksView.layout().metadata().addressOf_vector(),
              nestedBlocksView.secondLayout().metadata().addressOf_vector());

    ffx::soa::PortableHostCollection<GenericSoA> outputHost(ffx::host(), sizes[0], sizes[1], sizes[5]);
    alpaka::memcpy(queue, outputHost.buffer(), blocksCollection.buffer());
    alpaka::wait(queue);

    checkDataAndMetadata(h_nestedBlocksCollection.const_view(), outputHost.const_view());
  }
}

TEST_F(SoAPortableNestedBlocksAlpakaTest, DeepCopyConstViewHostToHostAndDeviceToDevice) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty())
    GTEST_SKIP();

  for (auto const& device : devices) {
    Queue queue(device);

    ffx::soa::PortableCollection<Device, NestedBlocksSoA> nestedBlocksCollection(queue, sizes);
    ffx::soa::PortableHostCollection<NestedBlocksSoA> h_nestedBlocksCollection(queue, sizes);
    SetUpNestedCollection(queue, nestedBlocksCollection, h_nestedBlocksCollection);

    NestedBlocksConstView nestedBlocksConstView = nestedBlocksCollection.const_view();

    GenericSoAConstView genericConstView{nestedBlocksConstView.firstBlocks(), nestedBlocksConstView.secondLayout()};

    // Verify metadata
    EXPECT_EQ(genericConstView.metadata().size()[0], sizes[0]);
    EXPECT_EQ(genericConstView.blocks().first().metadata().size(), sizes[0]);
    EXPECT_EQ(genericConstView.metadata().size()[1], sizes[1]);
    EXPECT_EQ(genericConstView.blocks().second().metadata().size(), sizes[1]);
    EXPECT_EQ(genericConstView.metadata().size()[2], sizes[5]);
    EXPECT_EQ(genericConstView.layout().metadata().size(), sizes[5]);

    // Check equality of memory addresses
    EXPECT_EQ(genericConstView.blocks().first().metadata().addressOf_column(),
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_column());
    EXPECT_EQ(genericConstView.blocks().first().metadata().addressOf_vector(),
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_vector());
    EXPECT_EQ(genericConstView.blocks().second().metadata().addressOf_column(),
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_column());
    EXPECT_EQ(genericConstView.blocks().second().metadata().addressOf_vector(),
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_vector());
    EXPECT_EQ(genericConstView.layout().metadata().addressOf_column(),
              nestedBlocksConstView.secondLayout().metadata().addressOf_column());
    EXPECT_EQ(genericConstView.layout().metadata().addressOf_vector(),
              nestedBlocksConstView.secondLayout().metadata().addressOf_vector());

    // PortableCollection hosting aggregated columns
    ffx::soa::PortableCollection<Device, GenericSoA> genericCollection(queue, sizes[0], sizes[1], sizes[5]);
    genericCollection.deepCopy(queue, genericConstView);

    GenericSoAConstView copiedGenericConstView = genericCollection.const_view();
    EXPECT_NE(copiedGenericConstView.blocks().first().metadata().addressOf_column(),
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.blocks().first().metadata().addressOf_vector(),
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_vector());
    EXPECT_NE(copiedGenericConstView.blocks().second().metadata().addressOf_column(),
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.blocks().second().metadata().addressOf_vector(),
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_vector());
    EXPECT_NE(copiedGenericConstView.layout().metadata().addressOf_column(),
              nestedBlocksConstView.secondLayout().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.layout().metadata().addressOf_vector(),
              nestedBlocksConstView.secondLayout().metadata().addressOf_vector());

    ffx::soa::PortableHostCollection<GenericSoA> outputHost(ffx::host(), sizes[0], sizes[1], sizes[5]);
    alpaka::memcpy(queue, outputHost.buffer(), genericCollection.buffer());
    alpaka::wait(queue);

    checkDataAndMetadata(h_nestedBlocksCollection.const_view(), outputHost.const_view());
  }
}

TEST_F(SoAPortableNestedBlocksAlpakaTest, DeepCopyConstViewDeviceToHost) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty())
    GTEST_SKIP();

  for (auto const& device : devices) {
    Queue queue(device);

    ffx::soa::PortableCollection<Device, NestedBlocksSoA> nestedBlocksCollection(queue, sizes);
    ffx::soa::PortableHostCollection<NestedBlocksSoA> h_nestedBlocksCollection(queue, sizes);
    SetUpNestedCollection(queue, nestedBlocksCollection, h_nestedBlocksCollection);

    NestedBlocksConstView nestedBlocksConstView = nestedBlocksCollection.const_view();

    GenericSoAConstView genericConstView{nestedBlocksConstView.firstBlocks(), nestedBlocksConstView.secondLayout()};

    // PortableHostCollection hosting aggregated columns
    ffx::soa::PortableHostCollection<GenericSoA> genericCollection(queue, sizes[0], sizes[1], sizes[5]);
    genericCollection.deepCopy(queue, genericConstView);
    alpaka::wait(queue);

    GenericSoAConstView copiedGenericConstView = genericCollection.const_view();
    EXPECT_NE(copiedGenericConstView.blocks().first().metadata().addressOf_column(),
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.blocks().first().metadata().addressOf_vector(),
              nestedBlocksConstView.firstBlocks().first().metadata().addressOf_vector());
    EXPECT_NE(copiedGenericConstView.blocks().second().metadata().addressOf_column(),
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.blocks().second().metadata().addressOf_vector(),
              nestedBlocksConstView.firstBlocks().second().metadata().addressOf_vector());
    EXPECT_NE(copiedGenericConstView.layout().metadata().addressOf_column(),
              nestedBlocksConstView.secondLayout().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.layout().metadata().addressOf_vector(),
              nestedBlocksConstView.secondLayout().metadata().addressOf_vector());

    checkDataAndMetadata(h_nestedBlocksCollection.const_view(), genericCollection.const_view());
  }
}

TEST_F(SoAPortableNestedBlocksAlpakaTest, DeepCopyConstViewHostToDevice) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty())
    GTEST_SKIP();

  for (auto const& device : devices) {
    Queue queue(device);

    ffx::soa::PortableCollection<Device, NestedBlocksSoA> nestedBlocksCollection(queue, sizes);
    ffx::soa::PortableHostCollection<NestedBlocksSoA> h_nestedBlocksCollection(queue, sizes);
    SetUpNestedCollection(queue, nestedBlocksCollection, h_nestedBlocksCollection);

    GenericSoAConstView genericConstView{h_nestedBlocksCollection.const_view().firstBlocks(),
                                         h_nestedBlocksCollection.const_view().secondLayout()};

    // PortableCollection hosting aggregated columns
    ffx::soa::PortableCollection<Device, GenericSoA> genericCollection(queue, sizes[0], sizes[1], sizes[5]);
    genericCollection.deepCopy(queue, genericConstView);
    alpaka::wait(queue);

    GenericSoAConstView copiedGenericConstView = genericCollection.const_view();
    EXPECT_NE(copiedGenericConstView.blocks().first().metadata().addressOf_column(),
              h_nestedBlocksCollection.const_view().firstBlocks().first().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.blocks().first().metadata().addressOf_vector(),
              h_nestedBlocksCollection.const_view().firstBlocks().first().metadata().addressOf_vector());
    EXPECT_NE(copiedGenericConstView.blocks().second().metadata().addressOf_column(),
              h_nestedBlocksCollection.const_view().firstBlocks().second().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.blocks().second().metadata().addressOf_vector(),
              h_nestedBlocksCollection.const_view().firstBlocks().second().metadata().addressOf_vector());
    EXPECT_NE(copiedGenericConstView.layout().metadata().addressOf_column(),
              h_nestedBlocksCollection.const_view().secondLayout().metadata().addressOf_column());
    EXPECT_NE(copiedGenericConstView.layout().metadata().addressOf_vector(),
              h_nestedBlocksCollection.const_view().secondLayout().metadata().addressOf_vector());

    ffx::soa::PortableHostCollection<GenericSoA> outputHost(ffx::host(), sizes[0], sizes[1], sizes[5]);
    alpaka::memcpy(queue, outputHost.buffer(), genericCollection.buffer());
    alpaka::wait(queue);

    checkDataAndMetadata(h_nestedBlocksCollection.const_view(), outputHost.const_view());
  }
}