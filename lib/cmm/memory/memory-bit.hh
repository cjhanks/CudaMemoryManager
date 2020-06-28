#ifndef CMM_MEMORY_BIT_HH_
#define CMM_MEMORY_BIT_HH_

#include <deque>
#include <memory>
#include <thread>
#include <unordered_map>

#include "cmm/memory.hh"
#include "cmm/memory/opaque-pointer.hh"
#include "cmm/tools/event.hh"
#include "cmm/tools/queue.hh"


namespace cmm { namespace bit {
///
/// Internal function to install which discretizer should be used globally.
///
void
InstallDiscretizer(std::unique_ptr<Discretizer>&& discretizer);
} // ns bit

///
/// @struct PinnedMemorySegment
///
struct PinnedMemorySegment {
  void* ptr_gpu;
  void* ptr_cpu;
};

///
/// @class PinnedMemorySegmentList
///
class PinnedMemorySegmentList {
 public:
  PinnedMemorySegment
  Next();

  void
  Return(void* ptr_gpu, void* ptr_cpu);

 private:
  std::deque<PinnedMemorySegment> segments;
  std::mutex mutex;
};

using GpuMemorySegment = void*;

///
/// @class GpuMemorySegmentList
///
class GpuMemorySegmentList {
 public:
  GpuMemorySegment
  Next();

  void
  Return(GpuMemorySegment ptr);

 private:
  std::deque<GpuMemorySegment> segments;
  std::mutex mutex;
};

///
/// @class MemoryManager
///
class MemoryManager {
 public:
  static MemoryManager&
  Instance();

  MemoryManager() = default;

  void
  Install(std::unique_ptr<Discretizer>&& discretizer);

  // {
  PinnedMemory
  NewPinned(std::size_t bytes);

  void
  Free(PinnedMemory& memory);
  // }

  // {
  GpuMemory
  NewGpu(std::size_t bytes);

  void
  Free(GpuMemory& memory);
  // }

 private:
  std::unique_ptr<Discretizer> discretizer;
  std::thread thread_handle;

  struct ReturnRecord {
    void*        ptr_gpu;
    void*        ptr_cpu;
    std::size_t  size;
    StreamEvent  event;
    cudaStream_t stream;
  };

  using PinMap = std::unordered_map<std::size_t, PinnedMemorySegmentList>;
  using GpuMap = std::unordered_map<std::size_t, GpuMemorySegmentList>;

  PinMap pin_memory;
  GpuMap gpu_memory;
  bit::RecvBlockingQueue<ReturnRecord> returns;

  /// Launches the thread, this should be called by Install(...)
  void
  Start();

  /// Permament runtime loop for the memory thread.
  void
  Loop();

  void
  ValidateOrThrow() const;
};

} // ns cmm

#endif // CMM_MEMORY_BIT_HH_
