#include <gtest/gtest.h>

#include "cmm/cmm.hh"

TEST(MemoryManager, Basic) {
  // Will throw, since cmm has not been initialized.
  EXPECT_THROW(cmm::MemoryManager::Instance(), cmm::Error);

  cmm::Specification spec;
  spec.discretizer = std::make_unique<cmm::DiscretizerNone>();
  cmm::Install(std::move(spec));

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
  } while (0);

  do {
    cmm::GpuMemory gm1(100);
    ASSERT_EQ(gm1.Size(), 100);
    ASSERT_NE(gm1.PointerGPU(), nullptr);

    //cmm::GpuMemory gm2 = std::move(gm1);
  } while (0);
}
