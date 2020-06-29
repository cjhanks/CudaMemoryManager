#ifndef CMM_OPAQUE_POINTER_HH_
#define CMM_OPAQUE_POINTER_HH_

#include <cstring>

namespace cmm {
class MemoryManager;

///
/// @class PinMemory
///
class PinMemory {
 public:
  PinMemory();
  PinMemory(std::size_t size);
  ~PinMemory();

  /// {
  PinMemory(PinMemory&& rhs);
  PinMemory&
  operator=(PinMemory&& rhs);

  PinMemory(const PinMemory&) = delete;
  PinMemory&
  operator=(const PinMemory&) = delete;
  /// }

  /// {
  const void*
  PointerGPU() const;

  void*
  PointerGPU();

  const void*
  PointerCPU() const;

  void*
  PointerCPU();
  /// }

  /// {
  void
  TransferToGPU(bool async=true);

  void
  TransferToCPU(bool async=false);
  /// }

  std::size_t
  Size() const;

 private:
  enum class Device {
    GPU,
    CPU,
  };

  Device device;
  bool   dirty;

  void* ptr_gpu;
  void* ptr_cpu;
  std::size_t size;

  friend class MemoryManager;
  PinMemory(void* ptr_gpu, void* ptr_cpu, std::size_t size);
};

///
/// @class GpuMemory
///
class GpuMemory {
 public:
  GpuMemory();
  GpuMemory(std::size_t size);
  ~GpuMemory();

  /// {
  GpuMemory(GpuMemory&& rhs);
  GpuMemory&
  operator=(GpuMemory&& rhs);

  GpuMemory(const GpuMemory&) = delete;
  GpuMemory&
  operator=(const GpuMemory&) = delete;
  /// }

  void
  Load(void* cpu_ptr);

  const void*
  PointerGPU() const;

  void*
  PointerGPU();

  std::size_t
  Size() const;

 private:
  void* ptr;
  std::size_t size;

  friend class MemoryManager;
  GpuMemory(void* ptr, std::size_t size);
};
} // ns cmm

#endif // CMM_POINTER_HH_
