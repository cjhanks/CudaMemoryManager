#include <gtest/gtest.h>

#include "cmm/cmm.hh"
#include "cmm/logging.hh"

TEST(MemoryManager, Basic) {
  // Will not throw.
  auto& instance = cmm::MemoryManager::Instance();
  (void) instance;

  // - Test the pinned memory.
  do {
    cmm::PinnedMemory pm1(100);
    ASSERT_EQ(pm1.Size(), 100);

    pm1.PointerCPU();
    EXPECT_THROW(pm1.PointerGPU(), cmm::Error);
    pm1.TransferToGPU();

    pm1.PointerGPU();
    EXPECT_THROW(pm1.PointerCPU(), cmm::Error);

    cmm::PinnedMemory pm2 = std::move(pm1);
    ASSERT_EQ(pm1.Size(), 0);
    ASSERT_EQ(pm2.Size(), 100);
  } while (0);

  // - Test the GPU memory.
  do {
    cmm::GpuMemory gm1(100);
    ASSERT_EQ(gm1.Size(), 100);
    ASSERT_NE(gm1.PointerGPU(), nullptr);

    cmm::GpuMemory gm2 = std::move(gm1);
    ASSERT_EQ(gm1.Size(), 0);
    ASSERT_EQ(gm2.Size(), 100);
  } while (0);

  do {
    struct ThisThing {
      unsigned a;
      unsigned b;
    };

    cmm::TypedPinnedMemory<ThisThing> ptr;
    ptr.PointerCPU()->a = 3;
    ptr.PointerCPU()->b = 4;
    ptr.TransferToGPU(false);

    ASSERT_EQ(ptr.PointerCPU()->a, 3);
    ASSERT_EQ(ptr.PointerCPU()->b, 4);
  } while (0);

}
