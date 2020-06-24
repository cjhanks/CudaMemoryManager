#include "event.hh"
#include "error.hh"

namespace cmm {
StreamEvent::StreamEvent()
  : event(nullptr)
{
  static constexpr unsigned Flags = cudaEventDisableTiming;
  Error::Check(cudaEventCreateWithFlags(&event, Flags));
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
