#ifndef CMM_MATRIX_MATRIX_HH_
#define CMM_MATRIX_MATRIX_HH_

#include "cmm/memory.hh"

namespace cmm {
///
/// @class Matrix
///
template <typename Type, std::size_t Dims, typename MemoryArray>
class Matrix {
 public:
  Matrix() = default;

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
    return memory.PointerGPU()[indexer.Index(args...)];
#else
    return memory.PointerCPU()[indexer.Index(args...)];
#endif
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

 private:
  Indexer<Dims> indexer;
  MemoryArray memory;
};

///
/// Alias for the pinned memory type.
///
template <typename Type, std::size_t Dims>
using PinnedMatrix = Matrix<Type, Dims, TypedPinnedMemory<Type>>;

///
/// Alias for the GPU memory type.
///
template <typename Type, std::size_t Dims>
using GpuMatrix = Matrix<Type, Dims, TypedGpuMemoryArray<Type>>;
} // ns cmm

#endif // CMM_MATRIX_MATRIX_HH_
