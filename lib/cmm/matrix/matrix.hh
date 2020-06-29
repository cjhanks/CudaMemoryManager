#ifndef CMM_MATRIX_MATRIX_HH_
#define CMM_MATRIX_MATRIX_HH_

#include "cmm/memory.hh"
#include "cmm/matrix/matrix-broadcast.hh"

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

  template <typename Operation, typename RhsType>
  void
  GpuBroadcast(Operation, RhsType value)
  {
    auto rhs = ShapedLike(*this);
  }

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

///
/// Alias for the GPU memory type.
///
template <typename Type, std::size_t Dims>
using GpuMatrix = Matrix<Type, Dims, TypedGpuMemoryArray<Type>>;
} // ns cmm

#endif // CMM_MATRIX_MATRIX_HH_
