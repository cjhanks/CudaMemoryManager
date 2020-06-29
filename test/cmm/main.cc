#include <gtest/gtest.h>

#include "cmm/cmm.hh"

int
main(int argc, char **argv)
{
  cmm::Specification spec;
  spec.discretizer = std::make_unique<cmm::DiscretizerNone>();
  cmm::Install(std::move(spec));

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
