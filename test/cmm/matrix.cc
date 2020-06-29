#include <gtest/gtest.h>

#include "cmm/matrix/index.hh"
#include "cmm/matrix/matrix.hh"

TEST(Indexer, Basic1) {
  cmm::Indexer<1> indexer0;
  cmm::Indexer<1> indexer1(100);

  ASSERT_EQ(indexer0.Size(),    0);
  ASSERT_EQ(indexer1.Size(),  100);
  ASSERT_EQ(indexer1.Size(0), 100);
}

TEST(Indexer, Basic2) {
  cmm::Indexer<2> indexer0;
  cmm::Indexer<2> indexer1(100, 50);

  ASSERT_EQ(indexer0.Size(),  0);
  ASSERT_EQ(indexer1.Size(),  50 * 100);
  ASSERT_EQ(indexer1.Size(0), 100);
  ASSERT_EQ(indexer1.Size(1), 50);

  std::size_t n = 0;
  for (std::size_t i = 0; i < indexer1.Size(0); ++i) {
    for (std::size_t j = 0; j < indexer1.Size(1); ++j) {
      ASSERT_EQ(indexer1.Index(i, j), n++);
    }
  }
}

TEST(Indexer, Basic3) {
  cmm::Indexer<3> indexer0;
  cmm::Indexer<3> indexer1(100, 50, 20);

  ASSERT_EQ(indexer0.Size(),  0);
  ASSERT_EQ(indexer1.Size(0), 100);
  ASSERT_EQ(indexer1.Size(1), 50);
  ASSERT_EQ(indexer1.Size(2), 20);
  ASSERT_EQ(indexer1.Size(),  50 * 100 * 20);

  std::size_t n = 0;
  for (std::size_t i = 0; i < indexer1.Size(0); ++i) {
    for (std::size_t j = 0; j < indexer1.Size(1); ++j) {
      for (std::size_t k = 0; k < indexer1.Size(2); ++k) {
        ASSERT_EQ(indexer1.Index(i, j, k), n++);
      }
    }
  }
}

TEST(Matrix, Basic) {
  cmm::PinMatrix<float, 2> p0;
  cmm::PinMatrix<float, 2> p1(100, 40);

  ASSERT_EQ(p0.Size(),  0);
  ASSERT_EQ(p1.Size(0), 100);
  ASSERT_EQ(p1.Size(1), 40);
  ASSERT_EQ(p1.Size(),  40 * 100);
}
