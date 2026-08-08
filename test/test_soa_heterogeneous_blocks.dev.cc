#include "../include/ffx/core/alpaka/config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <alpaka/alpaka.hpp>

#include "ffx/ffx.h"

using namespace ffx_runtime;

GENERATE_SOA_LAYOUT(NodesT, SOA_COLUMN(int, id), SOA_SCALAR(int, count))

using Nodes = NodesT<>;

GENERATE_SOA_LAYOUT(EdgesT, SOA_COLUMN(int, src), SOA_COLUMN(int, dst), SOA_COLUMN(float, cost), SOA_SCALAR(int, count))

using Edges = EdgesT<>;

GENERATE_SOA_BLOCKS(OneBlockTemplate, SOA_BLOCK(nodes, NodesT))
GENERATE_SOA_BLOCKS(GraphT, SOA_BLOCK(nodes, NodesT), SOA_BLOCK(edges, EdgesT))

using OneBlock = OneBlockTemplate<>;
using OneBlockView = OneBlock::View;
using OneBlockConstView = OneBlock::ConstView;

using Graph = GraphT<>;
using GraphView = Graph::View;
using GraphConstView = Graph::ConstView;

// Fill SoAs kernel
struct FillSoAs {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, Nodes::View nodes, Edges::View edges) const {
    const int N = static_cast<int>(nodes.metadata().size());
    const int E = static_cast<int>(edges.metadata().size());

    // Fill nodes with indices
    for (auto i : alpaka::uniformElements(acc, nodes.metadata().size())) {
      nodes[i].id() = static_cast<int>(i);
    }
    if (alpaka::oncePerGrid(acc)) {
      nodes.count() = N;
    }

    // Fill edges with deterministic values
    for (auto j : alpaka::uniformElements(acc, edges.metadata().size())) {
      int src = static_cast<int>(j % N);
      int dst = static_cast<int>((j * 7 + 3) % N);
      edges[j].src() = src;
      edges[j].dst() = dst;
      edges[j].cost() = 0.5f * static_cast<float>(src + dst);
    }
    if (alpaka::oncePerGrid(acc)) {
      edges.count() = E;
    }
  }
};

// Fill single-block SoABlocks kernel
struct FillOneBlockSoABlocks {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, OneBlockView blocksView) const {
    const int N = static_cast<int>(blocksView.nodes().metadata().size());

    for (auto i : alpaka::uniformElements(acc, blocksView.nodes().metadata().size())) {
      blocksView.nodes()[i].id() = static_cast<int>(i);
    }
    if (alpaka::oncePerGrid(acc)) {
      blocksView.nodes().count() = N;
    }
  }
};

// Fill graph SoABlocks kernel
struct FillBlocks {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, GraphView blocksView) const {
    const int N = static_cast<int>(blocksView.nodes().metadata().size());
    const int E = static_cast<int>(blocksView.edges().metadata().size());

    for (auto i : alpaka::uniformElements(acc, blocksView.nodes().metadata().size())) {
      blocksView.nodes()[i].id() = static_cast<int>(i);
    }
    if (alpaka::oncePerGrid(acc)) {
      blocksView.nodes().count() = N;
    }

    for (auto j : alpaka::uniformElements(acc, blocksView.edges().metadata().size())) {
      int src = static_cast<int>(j % N);
      int dst = static_cast<int>((j * 7 + 3) % N);
      blocksView.edges()[j].src() = src;
      blocksView.edges()[j].dst() = dst;
      blocksView.edges()[j].cost() = 0.5f * static_cast<float>(src + dst);
    }
    if (alpaka::oncePerGrid(acc)) {
      blocksView.edges().count() = E;
    }
  }
};

class SoAPortableGraphAlpakaTest : public ::testing::Test {
protected:
  static constexpr int N = 50;
  static constexpr int E = 120;
};

TEST_F(SoAPortableGraphAlpakaTest, MinimalGraphInHeterogeneousEnvironment) {
  auto const& devices = ffx::devices<Platform>();
  if (devices.empty()) {
    GTEST_SKIP() << "No devices available for the " << BOOST_PP_STRINGIZE(ffx_runtime) << " backend, skipping.";
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    // Portable Collections for SoAs
    ffx::soa::PortableCollection<Device, Nodes> nodesCollection(queue, N);
    ffx::soa::PortableCollection<Device, Edges> edgesCollection(queue, E);
    Nodes::View& nodesCollectionView = nodesCollection.view();
    Edges::View& edgesCollectionView = edgesCollection.view();

    // Portable Collection for SoABlocks
    ffx::soa::PortableCollection<Device, OneBlock> oneBlockCollection(queue, N);
    OneBlockView& oneBlockCollectionView = oneBlockCollection.view();

    ffx::soa::PortableCollection<Device, Graph> graphCollection(queue, N, E);
    GraphView& graphCollectionView = graphCollection.view();

    // Work division
    constexpr std::size_t blockSize = 256;
    const std::size_t numberOfBlocksOneBlockVersion = ffx::divide_up_by(N, blockSize);
    const auto workDivOneBlockVersion = ffx::make_workdiv<Acc1D>(numberOfBlocksOneBlockVersion, blockSize);

    const std::size_t maxElems = std::max<std::size_t>(N, E);
    const std::size_t numberOfBlocks = ffx::divide_up_by(maxElems, blockSize);
    const auto workDiv = ffx::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    // Fill execution
    alpaka::exec<Acc1D>(queue, workDiv, FillSoAs{}, nodesCollectionView, edgesCollectionView);
    alpaka::exec<Acc1D>(queue, workDivOneBlockVersion, FillOneBlockSoABlocks{}, oneBlockCollectionView);
    alpaka::exec<Acc1D>(queue, workDiv, FillBlocks{}, graphCollectionView);
    alpaka::wait(queue);

    // Check results on host
    ffx::soa::PortableHostCollection<Nodes> nodesHost(ffx::host(), N);
    ffx::soa::PortableHostCollection<Edges> edgesHost(ffx::host(), E);
    ffx::soa::PortableHostCollection<OneBlock> oneBlockHost(ffx::host(), N);
    ffx::soa::PortableHostCollection<Graph> graphHost(ffx::host(), N, E);

    alpaka::memcpy(queue, nodesHost.buffer(), nodesCollection.buffer());
    alpaka::memcpy(queue, edgesHost.buffer(), edgesCollection.buffer());
    alpaka::memcpy(queue, oneBlockHost.buffer(), oneBlockCollection.buffer());
    alpaka::memcpy(queue, graphHost.buffer(), graphCollection.buffer());
    alpaka::wait(queue);

    const Nodes::ConstView nodesHostView = nodesHost.const_view();
    const Edges::ConstView edgesHostView = edgesHost.const_view();
    const OneBlockConstView oneBlockHostView = oneBlockHost.const_view();
    const GraphConstView graphHostView = graphHost.const_view();

    // Verify Nodes
    EXPECT_EQ(graphHostView.nodes().count(), N);
    for (int i = 0; i < N; ++i) {
      EXPECT_EQ(oneBlockHostView.nodes()[i].id(), nodesHostView[i].id());
      EXPECT_EQ(oneBlockHostView.nodes()[i].id(), i);
      EXPECT_EQ(graphHostView.nodes()[i].id(), nodesHostView[i].id());
      EXPECT_EQ(graphHostView.nodes()[i].id(), i);
    }

    // Verify Edges
    EXPECT_EQ(graphHostView.edges().count(), E);
    for (int j = 0; j < E; ++j) {
      EXPECT_EQ(graphHostView.edges()[j].src(), edgesHostView[j].src());
      EXPECT_EQ(graphHostView.edges()[j].dst(), edgesHostView[j].dst());
      EXPECT_FLOAT_EQ(graphHostView.edges()[j].cost(), edgesHostView[j].cost());

      int src = j % N;
      int dst = (j * 7 + 3) % N;
      EXPECT_EQ(graphHostView.edges()[j].src(), src);
      EXPECT_EQ(graphHostView.edges()[j].dst(), dst);
      EXPECT_NEAR(graphHostView.edges()[j].cost(), 0.5f * static_cast<float>(src + dst), 1e-6f);
    }
  }
}