#include <type_traits>

#include <gtest/gtest.h>

#include "alpaka/alpaka.hpp"
#include "ffx/ffx.h"

namespace {

  struct TestStruct {
    int a;
    float b;
  };

}  // namespace

// Static assert verifying type aliased host object safety
static_assert(
    std::is_same_v<ffx::soa::PortableObject<alpaka::DevCpu, TestStruct>, ffx::soa::PortableHostObject<TestStruct>>,
    "PortableObject on CpuDevice must alias PortableHostObject");

TEST(PortableObjectHostTest, InitializeBySettingMembersWithDevice) {
  ffx::soa::PortableObject<alpaka::DevCpu, TestStruct> obj(ffx::host());
  obj->a = 42;

  EXPECT_EQ(obj->a, 42);
}

TEST(PortableObjectHostTest, InitializeBySettingMembersWithQueue) {
  alpaka::QueueCpuBlocking queue{ffx::host()};
  ffx::soa::PortableObject<alpaka::DevCpu, TestStruct> obj(queue);
  obj->a = 42;

  EXPECT_EQ(obj->a, 42);
}

TEST(PortableObjectHostTest, InitializeViaConstructorWithDevice) {
  ffx::soa::PortableObject<alpaka::DevCpu, TestStruct> obj(ffx::host(), TestStruct{42, 3.14f});

  EXPECT_EQ(obj->a, 42);
  EXPECT_FLOAT_EQ(obj->b, 3.14f);
}

TEST(PortableObjectHostTest, InitializeViaConstructorWithQueue) {
  alpaka::QueueCpuBlocking queue{ffx::host()};
  ffx::soa::PortableObject<alpaka::DevCpu, TestStruct> obj(queue, TestStruct{42, 3.14f});

  EXPECT_EQ(obj->a, 42);
  EXPECT_FLOAT_EQ(obj->b, 3.14f);
}