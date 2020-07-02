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
