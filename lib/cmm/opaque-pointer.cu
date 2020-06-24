#include "opaque-pointer.hh"

#include "cmm/assert.hh"

namespace cmm {
OpaquePointer::OpaquePointer(std::size_t size)
  : device(Device::CPU),
    dirty(false),
    ptr_gpu(nullptr),
    ptr_cpu(nullptr),
    size(size)
{
}

OpaquePointer::OpaquePointer(OpaquePointer&& rhs)
  : device(rhs.device),
    dirty(rhs.dirty),
    ptr_gpu(rhs.ptr_gpu),
    ptr_cpu(rhs.ptr_cpu),
    size(rhs.size)
{
  rhs.ptr_gpu = nullptr;
  rhs.ptr_cpu = nullptr;
}

OpaquePointer&
OpaquePointer::operator=(OpaquePointer&& rhs)
{
  device  = rhs.device;
  dirty   = rhs.dirty;
  ptr_gpu = rhs.ptr_gpu;
  ptr_cpu = rhs.ptr_cpu;
  size    = rhs.size;

  rhs.ptr_gpu = nullptr;
  rhs.ptr_cpu = nullptr;

  return *this;
}

const void*
OpaquePointer::PointerGPU() const
{
  CMM_ASSERT(device == Device::GPU || !dirty);
  return ptr_gpu;
}

void*
OpaquePointer::PointerGPU()
{
  CMM_ASSERT(device == Device::GPU || !dirty);
  dirty = true;
  return ptr_gpu;
}

const void*
OpaquePointer::PointerCPU() const
{
  CMM_ASSERT(device == Device::CPU || !dirty);
  return ptr_cpu;
}

void*
OpaquePointer::PointerCPU()
{
  CMM_ASSERT(device == Device::CPU || !dirty);
  dirty = true;
  return ptr_cpu;
}

void
OpaquePointer::TransferToGPU()
{
  if (device == Device::CPU && dirty) {
  }

  device = Device::GPU;
  dirty  = false;
}

void
OpaquePointer::TransferToCPU()
{
  if (device == Device::GPU && dirty) {
  }

  device = Device::CPU;
  dirty  = false;
}
} // ns cmm
