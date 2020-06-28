#ifndef CMM_TYPED_POINTER_HH_
#define CMM_TYPED_POINTER_HH_

#include "cmm/memory/opaque-pointer.hh"

namespace cmm {
///
/// @class  TypedPinnedMemory
/// @tparam Type to store.
///
template <typename Type>
class TypedPinnedMemory : public PinnedMemory {
 public:
  TypedPinnedMemory()
    : PinnedMemory(sizeof(Type)) {}

  const Type*
  PointerGPU() const
  { return (const Type*) PinnedMemory::PointerGPU(); }

  Type*
  PointerGPU()
  { return (Type*) PinnedMemory::PointerGPU(); }

  const Type*
  PointerCPU() const
  { return (const Type*) PinnedMemory::PointerCPU(); }

  Type*
  PointerCPU()
  { return (Type*) PinnedMemory::PointerCPU(); }
};

///
/// @class  TypedGpuMemory
/// @tparam Type to store.
///
template <typename Type>
class TypedGpuMemory : public GpuMemory {
 public:
  TypedGpuMemory()
    : GpuMemory(sizeof(Type)) {}

  TypedGpuMemory(Type data)
    : GpuMemory(sizeof(Type))
  {
    this->Load((void*) &data);
  }
};

///
/// @class  TypedGpuMemory
/// @tparam Type to store.
///
//template <typename Type>
//class TypedGpuMemory : public GpuMemory {
// public:
//  TypedGpuMemory()
//    : GpuMemory(sizeof(Type)) {}
//
//  TypedGpuMemory(Type data)
//    : GpuMemory(sizeof(Type))
//  {
//    this->Load((void*) &data);
//  }
//};
} // ns cmm

#endif // CMM_TYPED_POINTER_HH_
