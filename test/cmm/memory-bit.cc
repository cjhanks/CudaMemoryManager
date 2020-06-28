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
}
