#include "opaque-pointer.hh"

#include "cmm/logging.hh"
#include "cmm/memory.hh"
#include "cmm/tools/assert.hh"
#include "cmm/tools/error.hh"
#include "cmm/tools/stream.hh"


namespace cmm {
PinnedMemory::PinnedMemory()
  : device(Device::CPU),
    dirty(false),
    ptr_gpu(nullptr),
    ptr_cpu(nullptr),
    size(0)
{
}

PinnedMemory::PinnedMemory(std::size_t size)
  : PinnedMemory()
{
  *this = std::move(MemoryManager::Instance().NewPinned(size));
}

PinnedMemory::PinnedMemory(void* ptr_gpu, void* ptr_cpu, std::size_t size)
  : device(Device::CPU),
    dirty(false),
    ptr_gpu(ptr_gpu),
    ptr_cpu(ptr_cpu),
    size(size)
{
}

PinnedMemory::~PinnedMemory()
{
  if (size) {
    MemoryManager::Instance().Free(*this);
  }
}

PinnedMemory::PinnedMemory(PinnedMemory&& rhs)
  : device(rhs.device),
    dirty(rhs.dirty),
    ptr_gpu(rhs.ptr_gpu),
    ptr_cpu(rhs.ptr_cpu),
    size(rhs.size)
{
  rhs.ptr_gpu = nullptr;
  rhs.ptr_cpu = nullptr;
  rhs.size    = 0;
}

PinnedMemory&
PinnedMemory::operator=(PinnedMemory&& rhs)
{
  device  = rhs.device;
  dirty   = rhs.dirty;
  ptr_gpu = rhs.ptr_gpu;
  ptr_cpu = rhs.ptr_cpu;
  size    = rhs.size;

  rhs.ptr_gpu = nullptr;
  rhs.ptr_cpu = nullptr;
  rhs.size    = 0;

  return *this;
}

const void*
PinnedMemory::PointerGPU() const
{
  if (device != Device::GPU && dirty)
    throw Error("Requested GPU memory that is fresher on CPU");

  return ptr_gpu;
}

void*
PinnedMemory::PointerGPU()
{
  if (device != Device::GPU && dirty)
    throw Error("Requested GPU memory that is fresher on CPU");

  device = Device::GPU;
  dirty = true;
  return ptr_gpu;
}

const void*
PinnedMemory::PointerCPU() const
{
  if (device != Device::CPU && dirty)
    throw Error("Requested GPU memory that is fresher on CPU");

  return ptr_cpu;
}

void*
PinnedMemory::PointerCPU()
{
  if (device != Device::CPU && dirty)
    throw Error("Requested GPU memory that is fresher on CPU");

  device = Device::CPU;
  dirty = true;
  return ptr_cpu;
}

void
PinnedMemory::TransferToGPU(bool async)
{
  if (device == Device::CPU && dirty) {
    Error::Check(
        cudaMemcpyAsync(ptr_gpu,
                        ptr_cpu,
                        size,
                        cudaMemcpyDeviceToHost,
                        Stream::This()));
    if (!async)
      Stream::This().Synchronize();
  }

  device = Device::GPU;
  dirty  = false;
}

void
PinnedMemory::TransferToCPU(bool async)
{
  if (device == Device::GPU && dirty) {
    Error::Check(
        cudaMemcpyAsync(ptr_cpu,
                        ptr_gpu,
                        size,
                        cudaMemcpyHostToDevice,
                        Stream::This()));
    if (!async)
      Stream::This().Synchronize();
  }

  device = Device::CPU;
  dirty  = false;
}

std::size_t
PinnedMemory::Size() const
{ return size; }

// -------------------------------------------------------------------------- //

GpuMemory::GpuMemory()
  : ptr(nullptr),
    size(0)
{}

GpuMemory::~GpuMemory()
{
  if (size)
    MemoryManager::Instance().Free(*this);
}

GpuMemory::GpuMemory(std::size_t size)
  : GpuMemory()
{
  *this = std::move(MemoryManager::Instance().NewGpu(size));
}

GpuMemory::GpuMemory(GpuMemory&& rhs)
{
  *this = std::move(rhs);
}

GpuMemory&
GpuMemory::operator=(GpuMemory&& rhs)
{
  ptr = rhs.ptr;
  size = rhs.size;
  rhs.ptr = nullptr;
  rhs.size = 0;

  return *this;
}

GpuMemory::GpuMemory(void* ptr, std::size_t size)
  : ptr(ptr),
    size(size)
{}

void
GpuMemory::Load(void* ptr_cpu)
{
  Error::Check(
        cudaMemcpyAsync(ptr_cpu,
                        ptr,
                        size,
                        cudaMemcpyHostToDevice,
                        Stream::This()));
  Stream::This().Synchronize();
}

const void*
GpuMemory::PointerGPU() const
{ return ptr; }

void*
GpuMemory::PointerGPU()
{ return ptr; }

std::size_t
GpuMemory::Size() const
{ return size; }
} // ns cmm
