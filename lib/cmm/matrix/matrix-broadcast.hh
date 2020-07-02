#ifndef CMM_MATRIX_MATRIX_BROADCAST_HH_
#define CMM_MATRIX_MATRIX_BROADCAST_HH_

#include "cmm/macro.hh"

namespace cmm {
#ifdef __CUDACC__
template <typename Type>
struct BroadcastOp;

template <typename TypeLhs>
struct BroadcastEqu {
  template <typename TypeRhs>
  cmm_device
  static TypeLhs
  Op(const TypeLhs&, TypeRhs& rhs)
  { return rhs; }
};

template <typename TypeLhs>
struct BroadcastAdd {
  template <typename TypeRhs>
  cmm_device
  static TypeLhs
  Op(const TypeLhs& lhs, TypeRhs& rhs)
  { return lhs + rhs; }
};

template <typename TypeLhs>
struct BroadcastSub {
  template <typename TypeRhs>
  cmm_device
  static TypeLhs
  Op(const TypeLhs& lhs, TypeRhs& rhs)
  { return lhs - rhs; }
};

template <typename TypeLhs>
struct BroadcastDiv {
  template <typename TypeRhs>
  cmm_device
  static TypeLhs
  Op(const TypeLhs& lhs, TypeRhs& rhs)
  { return lhs / rhs; }
};

template <typename TypeLhs>
struct BroadcastMul {
  template <typename TypeRhs>
  cmm_device
  static TypeLhs
  Op(const TypeLhs& lhs, TypeRhs& rhs)
  { return lhs * rhs; }
};

namespace bit {
template <typename Broadcast, typename TypeLhs, typename TypeRhs>
cmm_global
void
BroadcastInPlace(std::size_t size, TypeLhs* lhs, const TypeRhs rhs)
{
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    lhs[idx] = Broadcast::Op(lhs[idx], rhs);
}

template <typename Broadcast, typename TypeLhs, typename TypeRhs>
cmm_global
void
BroadcastIntoRet(
    std::size_t size, TypeLhs* ret, TypeLhs* lhs, const TypeRhs rhs)
{
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    ret[idx] = Broadcast::Op(lhs[idx], rhs);
}

template <typename Broadcast, typename TypeLhs, typename TypeRhs>
cmm_global
void
OperatePointwise(
    std::size_t size, TypeLhs* ret, TypeLhs* lhs, TypeRhs* rhs)
{
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    ret[idx] = Broadcast::Op(lhs[idx], rhs[idx]);
}

template <typename Broadcast, typename TypeLhs>
cmm_global
void
BroadcastFunctor(std::size_t size, TypeLhs* data)
{
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = Broadcast::Op(data[idx]);
}
} // ns bit
#endif
} // ns cmm

#endif // CMM_MATRIX_MATRIX_BROADCAST_HH_
