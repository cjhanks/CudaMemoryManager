#include <gtest/gtest.h>

#include "cmm/cmm.hh"


TEST(Error, NoThrows) {
  EXPECT_NO_THROW(cmm::Error::Check(cudaSuccess));
}

TEST(Error, Throws) {
  EXPECT_THROW(cmm::Error::Check(cudaErrorMemoryAllocation), cmm::Error);
}
