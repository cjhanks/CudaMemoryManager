#include "event.hh"

#include <atomic>

#include "cmm/tools/error.hh"
#include "cmm/tools/stream.hh"

namespace cmm {
namespace {
std::atomic<std::uint64_t> IdCounter(0);
} // ns

StreamEvent::StreamEvent()
  : event(nullptr),
    id(IdCounter++)
{
  static constexpr unsigned Flags = cudaEventDisableTiming;
  Error::Check(cudaEventCreateWithFlags(&event, Flags));
  Error::Check(cudaEventRecord(event, Stream::This()));
}

StreamEvent::~StreamEvent()
{
  if (event)
    Error::Check(cudaEventDestroy(event));
}

StreamEvent::StreamEvent(StreamEvent&& rhs)
  : event(rhs.event)
{
  rhs.event = nullptr;
}

StreamEvent&
StreamEvent::operator=(StreamEvent&& rhs)
{
  event = rhs.event;
  rhs.event = nullptr;

  return *this;
}

std::uint64_t
StreamEvent::Id() const
{
  return id;
}

bool
StreamEvent::IsComplete() const
{
  cudaError_t rc = cudaEventQuery(event);
  if (rc == cudaSuccess)
    return true;
  else
  if (rc == cudaErrorNotReady)
    return false;

  Error::Check(rc); // should throw
  return false;
}
} // ns cmm
