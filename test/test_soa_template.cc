#include <cstdlib>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <tuple>

#include <gtest/gtest.h>

#include "ffx/ffx.h"

// Define simple SoA Layout
GENERATE_SOA_LAYOUT(
    SimpleLayoutTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z), SOA_COLUMN(float, t))

using SimpleLayout = SimpleLayoutTemplate<>;

// Test Fixture for sharing allocated SoA memory
class SimpleLayoutTest : public ::testing::Test {
protected:
  static constexpr std::size_t slSize = 10;

  void SetUp() override {
    const std::size_t slBufferSize = SimpleLayout::computeDataSize(slSize);
    slBuffer.reset(reinterpret_cast<std::byte*>(aligned_alloc(SimpleLayout::alignment, slBufferSize)));
    ASSERT_NE(slBuffer, nullptr);

    layout = std::make_unique<SimpleLayout>(slBuffer.get(), slSize);
  }

  std::unique_ptr<std::byte, decltype(&std::free)> slBuffer{nullptr, std::free};
  std::unique_ptr<SimpleLayout> layout;
};

TEST_F(SimpleLayoutTest, RowWideCopiesAndConstViewAccess) {
  SimpleLayout::View slv{*layout};
  SimpleLayout::ConstView slcv{*layout};

  auto slv0 = slv[0];
  slv0.x() = 1.0f;
  slv0.y() = 2.0f;
  slv0.z() = 3.0f;
  slv0.t() = 5.0f;

  // Fill up via row copy
  for (std::size_t i = 1; i < slv.metadata().size(); ++i) {
    auto slvi = slv[i];
    slvi = slv[i - 1];
    auto slvix = slvi.x();
    slvi.x() += slvi.y();
    slvi.y() += slvi.z();
    slvi.z() += slvi.t();
    slvi.t() += slvix;
  }

  // Verification via mutable and const views
  float x = 1.0f, y = 2.0f, z = 3.0f, t = 5.0f;
  for (std::size_t i = 0; i < slv.metadata().size(); ++i) {
    auto slvi = slv[i];
    auto slcvi = slcv[i];

    EXPECT_FLOAT_EQ(slvi.x(), x);
    EXPECT_FLOAT_EQ(slvi.y(), y);
    EXPECT_FLOAT_EQ(slvi.z(), z);
    EXPECT_FLOAT_EQ(slvi.t(), t);

    EXPECT_FLOAT_EQ(slcvi.x(), x);
    EXPECT_FLOAT_EQ(slcvi.y(), y);
    EXPECT_FLOAT_EQ(slcvi.z(), z);
    EXPECT_FLOAT_EQ(slcvi.t(), t);

    auto tx = x;
    x += y;
    y += z;
    z += t;
    t += tx;
  }
}

TEST_F(SimpleLayoutTest, RowInitializerAndRestrictDisabled) {
  using View =
      SimpleLayout::ViewTemplate<ffx::soa::restrict_qualify::disabled, ffx::soa::range_checking::default_value>;
  using ConstView =
      SimpleLayout::ConstViewTemplate<ffx::soa::restrict_qualify::disabled, ffx::soa::range_checking::default_value>;

  View slv{*layout};
  ConstView slcv{*layout};

  auto slv0 = slv[0];
  slv0 = {7.0f, 11.0f, 13.0f, 17.0f};

  // Fill up using row initialization from ConstView
  for (std::size_t i = 1; i < slv.metadata().size(); ++i) {
    auto slvi = slv[i];
    slvi = slcv[i - 1];
    auto slvix = slvi.x();
    slvi.x() += slvi.y();
    slvi.y() += slvi.z();
    slvi.z() += slvi.t();
    slvi.t() += slvix;
  }

  // Verification
  auto [x, y, z, t] = std::make_tuple(7.0f, 11.0f, 13.0f, 17.0f);
  for (std::size_t i = 0; i < slv.metadata().size(); ++i) {
    auto slvi = slv[i];
    auto slcvi = slcv[i];

    EXPECT_FLOAT_EQ(slvi.x(), x);
    EXPECT_FLOAT_EQ(slvi.y(), y);
    EXPECT_FLOAT_EQ(slvi.z(), z);
    EXPECT_FLOAT_EQ(slvi.t(), t);

    EXPECT_FLOAT_EQ(slcvi.x(), x);
    EXPECT_FLOAT_EQ(slcvi.y(), y);
    EXPECT_FLOAT_EQ(slcvi.z(), z);
    EXPECT_FLOAT_EQ(slcvi.t(), t);

    auto tx = x;
    x += y;
    y += z;
    z += t;
    t += tx;
  }
}

TEST_F(SimpleLayoutTest, RangeCheckingViewEnabled) {
  using View = SimpleLayout::ViewTemplate<ffx::soa::restrict_qualify::default_value, ffx::soa::range_checking::enabled>;
  View slv{*layout};

  int underflow = -1;
  int overflow = static_cast<int>(slv.metadata().size());

  // Check row accessor boundaries
  EXPECT_THROW(slv[underflow], std::out_of_range);
  EXPECT_THROW(slv[overflow], std::out_of_range);

  // Check column element accessor boundaries
  EXPECT_THROW(slv.x(underflow), std::out_of_range);
  EXPECT_THROW(slv.x(overflow), std::out_of_range);
}

TEST_F(SimpleLayoutTest, RangeCheckingConstViewEnabled) {
  using ConstView =
      SimpleLayout::ConstViewTemplate<ffx::soa::restrict_qualify::default_value, ffx::soa::range_checking::enabled>;
  ConstView slcv{*layout};

  int underflow = -1;
  int overflow = static_cast<int>(slcv.metadata().size());

  // Check row accessor boundaries
  EXPECT_THROW(slcv[underflow], std::out_of_range);
  EXPECT_THROW(slcv[overflow], std::out_of_range);

  // Check column element accessor boundaries
  EXPECT_THROW(slcv.x(underflow), std::out_of_range);
  EXPECT_THROW(slcv.x(overflow), std::out_of_range);
}

TEST_F(SimpleLayoutTest, RangeCheckingViewExtendedSourceLocation) {
  using View =
      SimpleLayout::ViewTemplate<ffx::soa::restrict_qualify::default_value, ffx::soa::range_checking::extended>;
  View slv{*layout};

  int underflow = -1;
  int overflow = static_cast<int>(slv.metadata().size());

  auto verify_source_location_exception = [](auto function_call) {
    try {
      function_call();
      FAIL() << "Expected std::out_of_range exception, but none was thrown.";
    } catch (const std::out_of_range& e) {
      std::string_view msg{e.what()};
      EXPECT_NE(msg.find("at file"), std::string_view::npos) << "Exception message did not contain 'at file': " << msg;
    }
  };

  verify_source_location_exception([&]() { slv[underflow]; });
  verify_source_location_exception([&]() { slv[overflow]; });
  verify_source_location_exception([&]() { slv.x(underflow); });
  verify_source_location_exception([&]() { slv.x(overflow); });
}

TEST_F(SimpleLayoutTest, RangeCheckingConstViewExtendedSourceLocation) {
  using ConstView =
      SimpleLayout::ConstViewTemplate<ffx::soa::restrict_qualify::default_value, ffx::soa::range_checking::extended>;
  ConstView slcv{*layout};

  int underflow = -1;
  int overflow = static_cast<int>(slcv.metadata().size());

  auto verify_source_location_exception = [](auto function_call) {
    try {
      function_call();
      FAIL() << "Expected std::out_of_range exception, but none was thrown.";
    } catch (const std::out_of_range& e) {
      std::string_view msg{e.what()};
      EXPECT_NE(msg.find("at file"), std::string_view::npos) << "Exception message did not contain 'at file': " << msg;
    }
  };

  verify_source_location_exception([&]() { slcv[underflow]; });
  verify_source_location_exception([&]() { slcv[overflow]; });
  verify_source_location_exception([&]() { slcv.x(underflow); });
  verify_source_location_exception([&]() { slcv.x(overflow); });
}