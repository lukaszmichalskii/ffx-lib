#include <cstdint>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ffx/ffx.h"

namespace {

  GENERATE_SOA_LAYOUT(TestLayout, SOA_COLUMN(double, x), SOA_COLUMN(int32_t, id))

  using TestSoA = TestLayout<>;

  GENERATE_SOA_BLOCKS(Blocks4Layout,
                      SOA_BLOCK(first, TestLayout),
                      SOA_BLOCK(second, TestLayout),
                      SOA_BLOCK(third, TestLayout),
                      SOA_BLOCK(fourth, TestLayout))

  using Blocks4 = Blocks4Layout<>;

  using TestCollection1 = ffx::soa::PortableHostCollection<TestSoA>;
  using TestCollection2 = ffx::soa::PortableHostCollection<Blocks4>;

}  // namespace

TEST(PortableCollectionSizeCastTest, ValidSignedValues) {
  EXPECT_NO_THROW([&] { TestCollection1 check(0); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(1); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int8_t>(1)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int16_t>(2)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int32_t>(3)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int64_t>(4)); }());

  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int_fast8_t>(5)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int_fast16_t>(6)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int_fast32_t>(7)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int_fast64_t>(8)); }());

  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int_least8_t>(9)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int_least16_t>(10)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int_least32_t>(11)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::int_least64_t>(12)); }());

  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed char>(13)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<short>(14)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<short int>(15)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed short>(16)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed short int>(17)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<int>(18)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed>(19)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<long>(20)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<long int>(21)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed long>(22)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed long int>(23)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<long long>(24)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<long long int>(25)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed long long>(26)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed long long int>(27)); }());

  EXPECT_NO_THROW(ffx::soa::detail::size_cast(std::numeric_limits<int>::max()));
}

TEST(PortableCollectionSizeCastTest, ValidUnsignedValues) {
  EXPECT_NO_THROW([&] { TestCollection1 check(0); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(1); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint8_t>(1)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint16_t>(2)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint32_t>(3)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint64_t>(4)); }());

  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint_fast8_t>(5)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint_fast16_t>(6)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint_fast32_t>(7)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint_fast64_t>(8)); }());

  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint_least8_t>(9)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint_least16_t>(10)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint_least32_t>(11)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<std::uint_least64_t>(12)); }());

  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned char>(13)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned short>(14)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned short int>(15)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned>(16)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned int>(17)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned long>(18)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned long int>(19)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned long long>(20)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned long long int>(21)); }());
}

TEST(PortableCollectionSizeCastTest, InvalidNegativeValues) {
  EXPECT_THROW([&] { TestCollection1 check(-1); }(), std::runtime_error);
  EXPECT_THROW([&] { TestCollection1 check(-42); }(), std::runtime_error);
  EXPECT_THROW([&] { TestCollection1 check(std::numeric_limits<int>::min()); }(), std::runtime_error);
  EXPECT_THROW([&] { TestCollection1 check(std::int8_t{-1}); }(), std::runtime_error);
  EXPECT_THROW([&] { TestCollection1 check(std::int16_t{-1}); }(), std::runtime_error);
  EXPECT_THROW([&] { TestCollection1 check(std::int64_t{-1}); }(), std::runtime_error);
}

TEST(PortableCollectionSizeCastTest, SignedValuesExceedingIntMaxRejected) {
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<long long>(std::numeric_limits<int>::max()) + 1); }());

  EXPECT_NO_THROW([&] { TestCollection1 check(std::numeric_limits<std::int64_t>::max()); }());
}

TEST(PortableCollectionSizeCastTest, CharacterTypes) {
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<signed char>(0)); }());
  EXPECT_THROW([&] { TestCollection1 check(static_cast<signed char>(-1)); }(), std::runtime_error);

  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned char>(0)); }());
  EXPECT_NO_THROW([&] { TestCollection1 check(static_cast<unsigned char>(255)); }());
}

TEST(PortableCollectionSizeCastTest, BlocksCollectionConstructors) {
  EXPECT_NO_THROW([&] {
    TestCollection2 check(static_cast<std::int8_t>(1),
                          static_cast<std::int16_t>(2),
                          static_cast<std::int32_t>(3),
                          static_cast<std::int64_t>(42));
  }());

  EXPECT_NO_THROW([&] {
    TestCollection2 check(static_cast<std::uint8_t>(1),
                          static_cast<std::uint16_t>(2),
                          static_cast<std::uint32_t>(3),
                          static_cast<std::uint64_t>(42));
  }());

  EXPECT_THROW(
      [&] {
        TestCollection2 check(static_cast<std::uint8_t>(1),
                              static_cast<std::uint16_t>(2),
                              static_cast<std::uint32_t>(3),
                              static_cast<std::int64_t>(-1));
      }(),
      std::runtime_error);
}