#ifndef CMM_MATRIX_MATRIX_HH_
#define CMM_MATRIX_MATRIX_HH_

#include "cmm/memory.hh"
#include "cmm/matrix/matrix-broadcast.hh"
#include "cmm/tools/schedule.hh"
#include "cmm/tools/stream.hh"

namespace cmm {
///
/// @class Matrix
///
template <typename Type_, std::size_t Dims_, typename MemoryArray_>
class Matrix {
 public:
  // {
  using Type = Type_;
  static constexpr std::size_t Dims = Dims_;
  using MemoryArray = MemoryArray_;
  using Self = Matrix<Type, Dims, MemoryArray>;
  // }

  template <typename RhsType>
  static Self
  ShapedLike(const RhsType& rhs)
  {
    return Self(rhs.GetIndexer());
  }

  Matrix() = default;
  Matrix(Indexer<Dims> indexer)
    : indexer(indexer),
      memory(indexer.Size())
  {}

  template <typename... Args>
  Matrix(Args... args)
    : indexer(args...),
      memory(indexer.Size())
  {
  }

  template <typename... Args>
  Type&
  At(Args... args)
  {
#ifdef __CUDA_ARCH__
    return GpuAt(args...);
#else
    return CpuAt(args...);
#endif
  }

  template <typename... Args>
  Type&
  GpuAt(Args... args)
  {
    return memory.PointerGPU()[indexer.Index(args...)];
  }

  template <typename... Args>
  Type&
  CpuAt(Args... args)
  {
    return memory.PointerCPU()[indexer.Index(args...)];
  }

  MemoryArray&
  Memory()
  { return memory; }

  std::size_t
  Size() const
  { return indexer.Size(); }

  std::size_t
  Size(std::size_t index) const
  { return indexer.Size(index); }

#if __CUDACC__
  template <typename Operation, typename RhsType>
  Self
  GpuBroadcast(Operation, RhsType value)
  {
    auto rhs = ShapedLike(*this);
    auto sched = Schedule1D::MaxThreads(Size());
    cmm::bit::BroadcastIntoRet<Operation, Type, RhsType>
        <<<sched.B(),
           sched.T(),
           0,
           Stream::This()>>>(
        Size(),
        rhs.Memory().PointerGPU(),
        this->Memory().PointerGPU(),
        value
    );

    return rhs;
  }

  template <typename RhsType>
  Self
  operator+(const RhsType& value)
  { return this->GpuBroadcast(cmm::BroadcastAdd<RhsType>(), value); }

  template <typename RhsType>
  Self
  operator-(const RhsType& value)
  { return this->GpuBroadcast(cmm::BroadcastSub<RhsType>(), value); }

  template <typename RhsType>
  Self
  operator*(const RhsType& value)
  { return this->GpuBroadcast(cmm::BroadcastMul<RhsType>(), value); }

  template <typename RhsType>
  Self
  operator/(const RhsType& value)
  { return this->GpuBroadcast(cmm::BroadcastDiv<RhsType>(), value); }

  template <typename Operation, typename RhsType>
  void
  GpuBroadcastInPlace(Operation, RhsType value)
  {
    auto sched = Schedule1D::MaxThreads(Size());
    cmm::bit::BroadcastInPlace<Operation, Type, RhsType>
        <<<sched.B(),
           sched.T(),
           0,
           Stream::This()>>>(
        Size(),
        this->Memory().PointerGPU(),
        value
    );
  }

  template <typename RhsType>
  Self&
  operator=(const RhsType& value)
  {
    this->GpuBroadcastInPlace(cmm::BroadcastEqu<RhsType>(), value);
    return *this;
  }

  template <typename RhsType>
  Self&
  operator+=(const RhsType& value)
  {
    this->GpuBroadcastInPlace(cmm::BroadcastAdd<RhsType>(), value);
    return *this;
  }

  template <typename RhsType>
  Self&
  operator-=(const RhsType& value)
  {
    this->GpuBroadcastInPlace(cmm::BroadcastSub<RhsType>(), value);
    return *this;
  }

  template <typename RhsType>
  Self&
  operator*=(const RhsType& value)
  {
    this->GpuBroadcastInPlace(cmm::BroadcastMul<RhsType>(), value);
    return *this;
  }

  template <typename RhsType>
  Self&
  operator/=(const RhsType& value)
  {
    this->GpuBroadcastInPlace(cmm::BroadcastDiv<RhsType>(), value);
    return *this;
  }
#endif

  Indexer<Dims>
  GetIndexer() const
  { return indexer; }

 private:
  cmm::Indexer<Dims> indexer;
  MemoryArray memory;
};

///
/// Alias for the pinned memory type.
///
template <typename Type, std::size_t Dims>
using PinMatrix = Matrix<Type, Dims, TypedPinMemoryArray<Type>>;

template <typename Type>
using PinVector = PinMatrix<Type, 1>;

///
/// Alias for the GPU memory type.
///
template <typename Type, std::size_t Dims>
using GpuMatrix = Matrix<Type, Dims, TypedGpuMemoryArray<Type>>;

template <typename Type>
using GpuVector = GpuMatrix<Type, 1>;
} // ns cmm

#endif // CMM_MATRIX_MATRIX_HH_
