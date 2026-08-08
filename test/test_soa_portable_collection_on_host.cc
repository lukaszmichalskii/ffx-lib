#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include "ffx/ffx.h"

namespace {

  GENERATE_SOA_LAYOUT(TestLayout, SOA_COLUMN(double, x), SOA_COLUMN(int32_t, id))

  using TestSoA = TestLayout<>;

}  // namespace

TEST(PortableCollectionHostTest, HostCollectionAliasAndMetadata) {
  constexpr std::size_t size = 10;

  // Construct PortableCollection on CPU host device queue
  ffx::soa::PortableCollection<ffx::DevHost, TestSoA> coll(ffx::host(), size);

  // Check metadata size accessor
  EXPECT_EQ(coll->metadata().size(), size);

  // Compile-time check verifying PortableCollection<CpuDevice, T> aliasing PortableHostCollection<T>
  static_assert(
      std::is_same_v<ffx::soa::PortableCollection<ffx::DevHost, TestSoA>, ffx::soa::PortableHostCollection<TestSoA>>,
      "PortableCollection on CpuDevice must alias PortableHostCollection");
}