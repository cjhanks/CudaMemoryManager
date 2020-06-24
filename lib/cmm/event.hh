#ifndef CMM_EVENT_HH_
#define CMM_EVENT_HH_

#include <cuda_runtime.h>


namespace cmm {
///
/// @class StreamEvent
///
/// Captures an event within a stream and encapsulates the underlying cuda
/// types.
///
class StreamEvent {
 public:
  StreamEvent();
  ~StreamEvent();

  StreamEvent(const StreamEvent&) = delete;
  StreamEvent&
  operator=(const StreamEvent&) = delete;

  StreamEvent(StreamEvent&&);
  StreamEvent&
  operator=(StreamEvent&&);

  bool
  IsComplete() const;

 private:
  cudaEvent_t event;
};
} // ns cmm

#endif // CMM_EVENT_HH_
