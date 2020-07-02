#include <gtest/gtest.h>

#include "cmm/cmm.hh"
#include "cmm/matrix/matrix-broadcast.hh"


TEST(MatrixCU, ExplicitCallingConvention) {
  cmm::PinVector<unsigned> pv(100);

  // Assignment test
  pv.GpuBroadcastInPlace(
      cmm::BroadcastEqu<unsigned>(),
      1);
  pv.Memory().TransferToCPU();
  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 1);
  }

  // Add test
  pv.Memory().TransferToGPU();
  pv.GpuBroadcastInPlace(
      cmm::BroadcastAdd<unsigned>(),
      10);
  pv.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 11);
  }

  // Sub test
  pv.Memory().TransferToGPU();
  pv.GpuBroadcastInPlace(
      cmm::BroadcastSub<unsigned>(),
      5);
  pv.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 6);
  }

  // Mul test
  pv.Memory().TransferToGPU();
  pv.GpuBroadcastInPlace(
      cmm::BroadcastMul<unsigned>(),
      4);
  pv.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 24);
  }

  // Div test
  pv.Memory().TransferToGPU();
  pv.GpuBroadcastInPlace(
      cmm::BroadcastDiv<unsigned>(),
      2);
  pv.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 12);
  }
}

TEST(MatrixCU, BroadcastInPlaceNormalConvention) {
  cmm::PinVector<unsigned> pv(100);

  // Assignment test
  pv = 1;
  pv.Memory().TransferToCPU();
  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 1);
  }

  // Add test
  pv.Memory().TransferToGPU();
  pv += 10;
  pv.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 11);
  }

  // Sub test
  pv.Memory().TransferToGPU();
  pv -= 5;
  pv.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 6);
  }

  // Mul test
  pv.Memory().TransferToGPU();
  pv *= 4;
  pv.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 24);
  }

  // Div test
  pv.Memory().TransferToGPU();
  pv /= 2;
  pv.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 12);
  }
}

TEST(MatrixCU, BroadcastNormalConvention) {
  cmm::PinVector<unsigned> pv(100);
  cmm::PinVector<unsigned> pr;

  // Assignment test
  pv = 1;
  pv.Memory().TransferToCPU();
  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pv.At(n), 1);
  }

  // Add test
  pv.Memory().TransferToGPU();
  pr = pv + 10;
  pr.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pr.At(n), 11);
  }

  pv = std::move(pr);

  // Sub test
  pv.Memory().TransferToGPU();
  pr = pv - 5;
  pr.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pr.At(n), 6);
  }

  pv = std::move(pr);

  // Mul test
  pv.Memory().TransferToGPU();
  pr = pv * 4;
  pr.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pr.At(n), 24);
  }

  pv = std::move(pr);

  // Div test
  pv.Memory().TransferToGPU();
  pr = pv / 2;
  pr.Memory().TransferToCPU();

  cmm::Stream::This().Synchronize();
  for (std::size_t n = 0; n < pv.Size(); ++n) {
    ASSERT_EQ(pr.At(n), 12);
  }
}

TEST(MatrixCU, Pointwise) {
  cmm::PinVector<unsigned> a0(100);
  cmm::PinVector<unsigned> a1(100);
  cmm::PinVector<unsigned> ret;

  a0 = 2;
  a1 = 100;

  // Add
  ret = (a0 + a1);
  ret.Memory().TransferToCPU();

  ASSERT_EQ(ret.Size(), 100);
  for (std::size_t n = 0; n < ret.Size(); ++n) {
    ASSERT_EQ(ret.At(n), 102);
  }

  // Sub
  ret = (a1 - a0);
  ret.Memory().TransferToCPU();

  ASSERT_EQ(ret.Size(), 100);
  for (std::size_t n = 0; n < ret.Size(); ++n) {
    ASSERT_EQ(ret.At(n), 98);
  }

  // Mul
  ret = (a1 * a0);
  ret.Memory().TransferToCPU();

  ASSERT_EQ(ret.Size(), 100);
  for (std::size_t n = 0; n < ret.Size(); ++n) {
    ASSERT_EQ(ret.At(n), 200);
  }

  // Div
  ret = (a1 / a0);
  ret.Memory().TransferToCPU();

  ASSERT_EQ(ret.Size(), 100);
  for (std::size_t n = 0; n < ret.Size(); ++n) {
    ASSERT_EQ(ret.At(n), 50);
  }
}

TEST(MatrixCU, PointwiseInPlace) {
  cmm::PinVector<unsigned> a0(100);
  cmm::PinVector<unsigned> a1(100);

  a0 = 2;
  a1 = 100;

  // Add
  a0.Memory().TransferToGPU();
  a0 += a1;
  a0.Memory().TransferToCPU();

  ASSERT_EQ(a0.Size(), 100);
  for (std::size_t n = 0; n < a0.Size(); ++n) {
    ASSERT_EQ(a0.At(n), 102);
  }

  // Sub
  a0.Memory().TransferToGPU();
  a0 -= a1;
  a0.Memory().TransferToCPU();

  ASSERT_EQ(a0.Size(), 100);
  for (std::size_t n = 0; n < a0.Size(); ++n) {
    ASSERT_EQ(a0.At(n), 2);
  }

  // Mul
  a0.Memory().TransferToGPU();
  a0 *= a1;
  a0.Memory().TransferToCPU();

  ASSERT_EQ(a0.Size(), 100);
  for (std::size_t n = 0; n < a0.Size(); ++n) {
    ASSERT_EQ(a0.At(n), 200);
  }

  // Div
  a0.Memory().TransferToGPU();
  a0 /= a1;
  a0.Memory().TransferToCPU();

  ASSERT_EQ(a0.Size(), 100);
  for (std::size_t n = 0; n < a0.Size(); ++n) {
    ASSERT_EQ(a0.At(n), 2);
  }
}

TEST(MatrixCU, Mat2D_FullTest) {
  // - Initialize
  cmm::PinMatrix<double, 2> a0(1024, 512);
  cmm::PinMatrix<double, 2> a1(1024, 512);
  cmm::GpuMatrix<double, 2> a4(1024, 512);

  a4 = 100.0;

  for (std::size_t i = 0; i < a0.Size(0); ++i) {
    for (std::size_t j = 0; j < a0.Size(1); ++j) {
      a0.At(i, j) = i + j;
      a1.At(i, j) = i - j;
    }
  }

  a0.Memory().TransferToGPU();
  a1.Memory().TransferToGPU();

  // - Let's perform various complicated operations.
  auto a2 = a0 + 32.0;

  a1 /= a2;
  a4 += a1;
  a2  = a0 * a1 - a4;
  a0 += a1 - a2;

  auto a3 = (a0 += 4.0) - a1;

  a0.Memory().TransferToCPU();
  a1.Memory().TransferToCPU();
  a2.Memory().TransferToCPU();
  a3.Memory().TransferToCPU();

  ASSERT_EQ(a0.Size(0), 1024);
  ASSERT_EQ(a0.Size(1),  512);

  for (std::size_t i = 0; i < a0.Size(0); ++i) {
    for (std::size_t j = 0; j < a0.Size(1); ++j) {
      double a0_ = i + j;
      double a1_ = i - j;
      double a2_ = a0_ + 32.0;

      a1_ /= a2_;
      double a4_ = 100.0 + a1_;
      a2_  = a0_ * a1_ - a4_;
      a0_ += a1_ - a2_;

      double a3_ = (a0_ += 4.0) - a1_;

      ASSERT_NEAR(a0_, a0.At(i, j), 1e-9);
      ASSERT_NEAR(a1_, a1.At(i, j), 1e-9);
      ASSERT_NEAR(a2_, a2.At(i, j), 1e-9);
      ASSERT_NEAR(a3_, a3.At(i, j), 1e-9);
    }
  }
}
