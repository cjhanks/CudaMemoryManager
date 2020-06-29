#ifndef CMM_MEMORY_MANAGER_HH_
#define CMM_MEMORY_MANAGER_HH_

#include <deque>
#include <memory>
#include <mutex>
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
/// @struct PinMemorySegment
///
struct PinMemorySegment {
  void* ptr_gpu;
  void* ptr_cpu;
};

///
/// @class PinMemorySegmentList
///
class PinMemorySegmentList {
 public:
  PinMemorySegmentList(std::size_t size);

  PinMemorySegmentList(PinMemorySegmentList&&);
  PinMemorySegmentList&
  operator=(PinMemorySegmentList&&);

  PinMemorySegment
  Next();

  void
  Return(void* ptr_gpu, void* ptr_cpu);

 private:
  std::size_t size;
  std::deque<PinMemorySegment> segments;
  std::mutex mutex;
};

using GpuMemorySegment = void*;

///
/// @class GpuMemorySegmentList
///
class GpuMemorySegmentList {
 public:
  GpuMemorySegmentList(std::size_t size);

  GpuMemorySegmentList(GpuMemorySegmentList&&);
  GpuMemorySegmentList&
  operator=(GpuMemorySegmentList&&);

  GpuMemorySegment
  Next();

  void
  Return(GpuMemorySegment ptr);

 private:
  std::size_t size;
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

  MemoryManager();
  ~MemoryManager();

  void
  Install(std::unique_ptr<Discretizer>&& discretizer);

  // {
  PinMemory
  NewPin(std::size_t bytes);

  void
  Free(PinMemory& memory);
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
  std::atomic<bool> terminated;

  struct ReturnRecord {
    void*        ptr_gpu;
    void*        ptr_cpu;
    std::size_t  size;
    StreamEvent  event;
    cudaStream_t stream;
  };

  using PinMap = std::unordered_map<std::size_t, PinMemorySegmentList>;
  PinMap     pin_memory;
  std::mutex pin_memory_lock;

  using GpuMap = std::unordered_map<std::size_t, GpuMemorySegmentList>;
  GpuMap     gpu_memory;
  std::mutex gpu_memory_lock;

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

#endif // CMM_MEMORY_MANAGER_HH_
