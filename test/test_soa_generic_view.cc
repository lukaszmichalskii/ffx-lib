#include <cstdlib>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "ffx/ffx.h"

// Define SoA Layouts explicitly specifying 128-byte alignment
GENERATE_SOA_LAYOUT(SoAPositionTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_SCALAR(int, detectorType))

using SoAPosition = SoAPositionTemplate<128>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

GENERATE_SOA_LAYOUT(SoAPCATemplate,
                    SOA_COLUMN(float, eigenvector_1),
                    SOA_COLUMN(float, eigenvector_2),
                    SOA_COLUMN(float, eigenvector_3),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

using SoAPCA = SoAPCATemplate<128>;
using SoAPCAView = SoAPCA::View;
using SoAPCAConstView = SoAPCA::ConstView;

GENERATE_SOA_LAYOUT(GenericSoATemplate,
                    SOA_COLUMN(float, xPos),
                    SOA_COLUMN(float, yPos),
                    SOA_COLUMN(float, zPos),
                    SOA_EIGEN_COLUMN(Eigen::Vector3d, candidateDirection))

// Explicitly use 128 bytes to match CMSSW behavior
using GenericSoA = GenericSoATemplate<128>;
using GenericSoAView = GenericSoA::View;
using GenericSoAConstView = GenericSoA::ConstView;

// Test Fixture for Generic View Tests
class SoAGenericViewTest : public ::testing::Test {
protected:
  static constexpr std::size_t elems = 16;

  void SetUp() override {
    const std::size_t positionBufferSize = SoAPosition::computeDataSize(elems);
    const std::size_t pcaBufferSize = SoAPCA::computeDataSize(elems);

    bufferPos.reset(reinterpret_cast<std::byte*>(aligned_alloc(SoAPosition::alignment, positionBufferSize)));
    bufferPCA.reset(reinterpret_cast<std::byte*>(aligned_alloc(SoAPCA::alignment, pcaBufferSize)));

    position = std::make_unique<SoAPosition>(bufferPos.get(), elems);
    pca = std::make_unique<SoAPCA>(bufferPCA.get(), elems);

    positionView = std::make_unique<SoAPositionView>(*position);
    positionConstView = std::make_unique<SoAPositionConstView>(*position);
    pcaView = std::make_unique<SoAPCAView>(*pca);
    pcaConstView = std::make_unique<SoAPCAConstView>(*pca);

    // Populate data
    for (std::size_t i = 0; i < elems; ++i) {
      (*positionView)[i] = {i * 1.0f, i * 2.0f, i * 3.0f};
    }
    positionView->detectorType() = 1;

    float time = 0.01f;
    for (std::size_t i = 0; i < elems; ++i) {
      (*pcaView)[i].eigenvector_1() = (*positionView)[i].x() / time;
      (*pcaView)[i].eigenvector_2() = (*positionView)[i].y() / time;
      (*pcaView)[i].eigenvector_3() = (*positionView)[i].z() / time;
      (*pcaView)[i].candidateDirection()(0) = (*positionView)[i].x() / time;
      (*pcaView)[i].candidateDirection()(1) = (*positionView)[i].y() / time;
      (*pcaView)[i].candidateDirection()(2) = (*positionView)[i].z() / time;
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

TEST_F(SoAGenericViewTest, GenericViewPointersAndMutability) {
  const auto posRecs = positionView->records();
  const auto pcaRecs = pcaView->records();

  // Building GenericView from subsets of distinct SoA columns
  GenericSoAView genericView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

  // Check equality of memory addresses
  EXPECT_EQ(genericView.metadata().addressOf_xPos(), positionView->metadata().addressOf_x());
  EXPECT_EQ(genericView.metadata().addressOf_yPos(), positionView->metadata().addressOf_y());
  EXPECT_EQ(genericView.metadata().addressOf_zPos(), positionView->metadata().addressOf_z());
  EXPECT_EQ(genericView.metadata().addressOf_candidateDirection(), pcaView->metadata().addressOf_candidateDirection());

  // Check reference to original underlying memory
  genericView[3].xPos() = 0.0f;
  EXPECT_FLOAT_EQ(genericView[3].xPos(), (*positionConstView)[3].x());
}

TEST_F(SoAGenericViewTest, GenericConstViewPointers) {
  const auto posRecs = positionConstView->records();
  const auto pcaRecs = pcaConstView->records();

  GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

  EXPECT_EQ(genericConstView.metadata().addressOf_xPos(), positionConstView->metadata().addressOf_x());
  EXPECT_EQ(genericConstView.metadata().addressOf_yPos(), positionConstView->metadata().addressOf_y());
  EXPECT_EQ(genericConstView.metadata().addressOf_zPos(), positionConstView->metadata().addressOf_z());
  EXPECT_EQ(genericConstView.metadata().addressOf_candidateDirection(),
            pcaConstView->metadata().addressOf_candidateDirection());
}

TEST_F(SoAGenericViewTest, GenericConstViewFromMutableViews) {
  const auto posRecs = positionView->records();
  const auto pcaRecs = pcaView->records();

  GenericSoAConstView genericConstView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

  // Mutating original View reflects in GenericConstView
  (*positionView)[3].x() = 0.0f;
  EXPECT_FLOAT_EQ(genericConstView[3].xPos(), (*positionView)[3].x());
}

TEST_F(SoAGenericViewTest, DeepCopyGenericView) {
  const std::size_t genericBufferSize = GenericSoA::computeDataSize(elems);
  std::unique_ptr<std::byte, decltype(&std::free)> bufferGeneric{
      reinterpret_cast<std::byte*>(aligned_alloc(GenericSoA::alignment, genericBufferSize)), std::free};
  GenericSoA genericSoA(bufferGeneric.get(), elems);

  const auto posRecs = positionView->records();
  const auto pcaRecs = pcaView->records();
  GenericSoAView genericView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());

  // Perform deep copy
  genericSoA.deepCopy(genericView);
  GenericSoAView genericSoAView{genericSoA};

  // Check inequality of memory addresses (standalone buffer allocated)
  EXPECT_NE(genericSoAView.metadata().addressOf_xPos(), positionConstView->metadata().addressOf_x());
  EXPECT_NE(genericSoAView.metadata().addressOf_yPos(), positionConstView->metadata().addressOf_y());
  EXPECT_NE(genericSoAView.metadata().addressOf_zPos(), positionConstView->metadata().addressOf_z());
  EXPECT_NE(genericSoAView.metadata().addressOf_candidateDirection(),
            pcaConstView->metadata().addressOf_candidateDirection());

  // Check column alignments
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(genericSoAView.metadata().addressOf_xPos()) % GenericSoA::alignment);
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(genericSoAView.metadata().addressOf_yPos()) % GenericSoA::alignment);
  EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(genericSoAView.metadata().addressOf_zPos()) % GenericSoA::alignment);
  EXPECT_EQ(0u,
            reinterpret_cast<std::uintptr_t>(genericSoAView.metadata().addressOf_candidateDirection()) %
                GenericSoA::alignment);

  // Check contiguity of aligned columns
  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoAView.metadata().addressOf_xPos()) +
                ffx::soa::alignSize(elems * sizeof(float), GenericSoA::alignment),
            reinterpret_cast<std::byte*>(genericSoAView.metadata().addressOf_yPos()));
  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoAView.metadata().addressOf_yPos()) +
                ffx::soa::alignSize(elems * sizeof(float), GenericSoA::alignment),
            reinterpret_cast<std::byte*>(genericSoAView.metadata().addressOf_zPos()));
  EXPECT_EQ(reinterpret_cast<std::byte*>(genericSoAView.metadata().addressOf_zPos()) +
                ffx::soa::alignSize(elems * sizeof(float), GenericSoA::alignment),
            reinterpret_cast<std::byte*>(genericSoAView.metadata().addressOf_candidateDirection()));

  // Verify copied values
  for (std::size_t i = 0; i < elems; ++i) {
    EXPECT_FLOAT_EQ(genericSoAView[i].xPos(), (*positionConstView)[i].x());
    EXPECT_FLOAT_EQ(genericSoAView[i].yPos(), (*positionConstView)[i].y());
    EXPECT_FLOAT_EQ(genericSoAView[i].zPos(), (*positionConstView)[i].z());
    EXPECT_DOUBLE_EQ(genericSoAView[i].candidateDirection()(0), (*pcaConstView)[i].candidateDirection()(0));
    EXPECT_DOUBLE_EQ(genericSoAView[i].candidateDirection()(1), (*pcaConstView)[i].candidateDirection()(1));
    EXPECT_DOUBLE_EQ(genericSoAView[i].candidateDirection()(2), (*pcaConstView)[i].candidateDirection()(2));
  }

  // Check independence of copied SoA
  genericSoAView[3].xPos() = 0.0f;
  EXPECT_NE(genericSoAView[3].xPos(), (*positionView)[3].x());
}

// Standalone test for stride mismatch validation across differing alignments
TEST(SoAGenericViewExceptionTest, MismatchedEigenStrideThrowsException) {
  // Use 64-byte alignment for source layout
  using SoAPCA64 = SoAPCATemplate<64>;
  using SoAPCA64View = SoAPCA64::View;

  const std::size_t elems = 17;  // 17 elements * 8 bytes = 136 bytes
                                 // 136 aligned to 64 bytes = 192 (stride 24)
                                 // 136 aligned to 128 bytes = 256 (stride 32) -> mismatch!

  const std::size_t positionBufferSize = SoAPosition::computeDataSize(elems);
  const std::size_t pcaBufferSize = SoAPCA64::computeDataSize(elems);

  std::unique_ptr<std::byte, decltype(&std::free)> bufferPos{
      reinterpret_cast<std::byte*>(aligned_alloc(SoAPosition::alignment, positionBufferSize)), std::free};
  std::unique_ptr<std::byte, decltype(&std::free)> bufferPCA{
      reinterpret_cast<std::byte*>(aligned_alloc(SoAPCA64::alignment, pcaBufferSize)), std::free};

  SoAPosition position{bufferPos.get(), elems};
  SoAPCA64 pca{bufferPCA.get(), elems};

  SoAPositionView positionView{position};
  SoAPCA64View pcaView{pca};

  const auto posRecs = positionView.records();
  const auto pcaRecs = pcaView.records();

  // Validate that 128-byte aligned View throws runtime exception when constructed from 64-byte aligned columns
  try {
    GenericSoAView(posRecs.x(), posRecs.y(), posRecs.z(), pcaRecs.candidateDirection());
    FAIL() << "Expected std::runtime_error was not thrown.";
  } catch (const std::runtime_error& e) {
    std::string_view msg{e.what()};
    EXPECT_NE(msg.find("In constructor by column pointers: stride not equal between eigen columns: candidateDirection"),
              std::string_view::npos)
        << "Unexpected error message: " << msg;
  }
}