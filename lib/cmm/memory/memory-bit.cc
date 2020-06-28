#include "memory-bit.hh"

#include <functional>

#include "cmm/tools/error.hh"
#include "cmm/tools/stream.hh"


namespace cmm {
// -------------------------------------------------------------------------- //

GpuMemorySegment
GpuMemorySegmentList::Next()
{
  std::lock_guard<std::mutex> lock(mutex);
  auto elem = segments.front();
  segments.pop_front();
  return elem;
}

void
GpuMemorySegmentList::Return(GpuMemorySegment ptr_gpu)
{
  std::lock_guard<std::mutex> lock(mutex);
  segments.emplace_back(ptr_gpu);
}

// -------------------------------------------------------------------------- //

PinnedMemorySegment
PinnedMemorySegmentList::Next()
{
  std::lock_guard<std::mutex> lock(mutex);
  auto elem = segments.front();
  segments.pop_front();
  return elem;
}

void
PinnedMemorySegmentList::Return(void* ptr_gpu, void* ptr_cpu)
{
  std::lock_guard<std::mutex> lock(mutex);
  segments.emplace_back(
      (PinnedMemorySegment) {.ptr_gpu = ptr_gpu,
                             .ptr_cpu = ptr_cpu}
  );
}

// -------------------------------------------------------------------------- //

namespace {
MemoryManager mm;
} // ns

MemoryManager&
MemoryManager::Instance()
{
  mm.ValidateOrThrow();
  return mm;
}

namespace bit {
void
InstallDiscretizer(std::unique_ptr<Discretizer>&& discretizer)
{
  mm.Install(std::move(discretizer));
}
} // ns bit

void
MemoryManager::Install(std::unique_ptr<Discretizer>&& discretizer)
{
  this->discretizer = std::move(discretizer);
  Start();
}

void
MemoryManager::Start()
{
  thread_handle =
      std::move(std::thread(std::bind(&MemoryManager::Loop, this)));
  thread_handle.detach();
}

PinnedMemory
MemoryManager::NewPinned(std::size_t bytes)
{
  std::size_t size = discretizer->Compute(bytes);
  auto segment = pin_memory[size].Next();

  return PinnedMemory(segment.ptr_gpu,
                      segment.ptr_cpu,
                      bytes);
}

void
MemoryManager::Free(PinnedMemory& memory)
{
  ReturnRecord record;
  record.ptr_gpu = memory.ptr_gpu;
  record.ptr_cpu = memory.ptr_cpu;
  record.size    = discretizer->Compute(memory.Size());
  record.event   = StreamEvent::Create();
  record.stream  = Stream::This();
  returns.Push(std::move(record));
}

GpuMemory
MemoryManager::NewGpu(std::size_t bytes)
{
  std::size_t size = discretizer->Compute(bytes);
  auto segment = gpu_memory[size].Next();
  return GpuMemory(segment, bytes);
}

void
MemoryManager::Free(GpuMemory& memory)
{
  ReturnRecord record;
  record.ptr_gpu = memory.ptr;
  record.ptr_cpu = nullptr;
  record.size    = discretizer->Compute(memory.Size());
  record.event   = StreamEvent::Create();
  record.stream  = Stream::This();
  returns.Push(std::move(record));
}


void
MemoryManager::Loop()
{
  using RecordList = std::deque<ReturnRecord>;
  using RecordMap = std::unordered_map<cudaStream_t, RecordList>;

  RecordMap record_map;
  do {
    // Look for records which should be returned to the pool and queue them in
    // their respective stream queue.
    ReturnRecord record;
    while (returns.Pop(record, 0.01))
      record_map[record.stream].emplace_back(std::move(record));

    // Return memory to the pool.
    for (auto& record: record_map) {
      auto& list = record.second;
      while (list.size()
          && list.front().event.IsComplete()) {
        // This element is ready to be returned to the memory pool.
        auto& elem = list.front();

        if (elem.ptr_cpu) {
          pin_memory[elem.size].Return(elem.ptr_gpu, elem.ptr_cpu);
        } else {
          gpu_memory[elem.size].Return(elem.ptr_gpu);
        }

        list.pop_front();
      }
    }
  } while (true);
}

void
MemoryManager::ValidateOrThrow() const
{
  if (!discretizer)
    throw Error("Discretizer not set on memory manager");
}
} // ns cmm
