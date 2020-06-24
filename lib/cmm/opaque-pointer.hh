#ifndef CMM_OPAQUE_POINTER_HH_
#define CMM_OPAQUE_POINTER_HH_

#include <cstring>

namespace cmm {
///
/// @class OpaquePointer
///
/// This structure is at the base of all of pointer types.  It designed to track
/// the current state of the pointer, whether it resides on the GPU or CPU.
///
class OpaquePointer {
 public:
  explicit OpaquePointer(std::size_t size);
  ~OpaquePointer() = default;

  /// {
  OpaquePointer(OpaquePointer&& rhs);
  OpaquePointer&
  operator=(OpaquePointer&& rhs);

  OpaquePointer(const OpaquePointer&) = delete;
  OpaquePointer&
  operator=(const OpaquePointer&) = delete;
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
  TransferToGPU();

  void
  TransferToCPU();
  /// }

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
};
} // ns cmm

#endif // CMM_POINTER_HH_
