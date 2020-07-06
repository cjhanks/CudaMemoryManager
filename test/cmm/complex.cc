#include <gtest/gtest.h>

#include <complex>
#include <random>

#include "cmm/complex.hh"
#include "cmm/logging.hh"


template <typename Type>
void
ThisTest(Type epsilon)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::exponential_distribution<double> d(1);

  for (std::size_t n = 0; n < 100000; ++n) {
    std::complex<Type> s0(10 * d(gen), 10 * d(gen));
    cmm::Complex<Type> c0 = s0;

    ASSERT_NEAR(s0.real(), c0.real(), epsilon);
    ASSERT_NEAR(s0.imag(), c0.imag(), epsilon);
    ASSERT_NEAR(std::norm(s0),
                cmm::norm(c0), epsilon);
    ASSERT_NEAR(std::arg(s0),
                cmm::arg(c0), epsilon);
  }
}

TEST(Complex, Float32) {
  ThisTest<float>(1e-3);
}

TEST(Complex, Float64) {
  ThisTest<double>(1e-9);
}
