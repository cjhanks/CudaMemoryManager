#include <gtest/gtest.h>

#include "cmm/error.hh"
#include "cmm/event.hh"


TEST(Event, Basic) {
  cmm::StreamEvent ev0;
  ASSERT_TRUE(ev0.IsComplete());

  cmm::StreamEvent ev1 = std::move(ev0);
  ASSERT_TRUE(ev1.IsComplete());

  // Ev0 should be invalidated because of move.
  EXPECT_THROW(ev0.IsComplete(), cmm::Error);
}
