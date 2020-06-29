#ifndef CMM_TYPED_POINTER_HH_
#define CMM_TYPED_POINTER_HH_

#include "cmm/memory/opaque-pointer.hh"

namespace cmm {
///
/// @class  TypedPinMemory
/// @tparam Type to store.
///
template <typename Type>
class TypedPinMemory : public PinMemory {
 public:
  TypedPinMemory()
    : PinMemory(sizeof(Type)) {}

  const Type*
  PointerGPU() const
  { return (const Type*) PinMemory::PointerGPU(); }

  Type*
  PointerGPU()
  { return (Type*) PinMemory::PointerGPU(); }

  const Type*
  PointerCPU() const
  { return (const Type*) PinMemory::PointerCPU(); }

  Type*
  PointerCPU()
  { return (Type*) PinMemory::PointerCPU(); }
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

template <typename Type>
class TypedPinMemoryArray : public PinMemory {
 public:
  TypedPinMemoryArray()
    : PinMemory() {}

  TypedPinMemoryArray(std::size_t size)
    : PinMemory(sizeof(Type) * size),
      size(size)
  {}

  const Type*
  PointerGPU() const
  { return (const Type*) PinMemory::PointerGPU(); }

  Type*
  PointerGPU()
  { return (Type*) PinMemory::PointerGPU(); }

  const Type*
  PointerCPU() const
  { return (const Type*) PinMemory::PointerCPU(); }

  Type*
  PointerCPU()
  { return (Type*) PinMemory::PointerCPU(); }

  std::size_t
  Size() const
  { return size; }

 private:
  std::size_t size;
};

///
/// @class  TypedGpuMemory
/// @tparam Type to store.
///
template <typename Type>
class TypedGpuMemoryArray : public GpuMemory {
 public:
  TypedGpuMemoryArray()
    : GpuMemory() {}

  TypedGpuMemoryArray(std::size_t size)
    : GpuMemory(sizeof(Type) * size) {}

  const Type*
  PointerGPU() const
  { return (const Type*) GpuMemory::PointerGPU(); }

  Type*
  PointerGPU()
  { return (Type*) GpuMemory::PointerGPU(); }

  std::size_t
  Size() const
  { return size; }

 private:
  std::size_t size;
};
} // ns cmm

#endif // CMM_TYPED_POINTER_HH_
