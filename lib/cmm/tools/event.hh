#ifndef CMM_EVENT_HH_
#define CMM_EVENT_HH_

#include <cstdint>
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
  static StreamEvent
  Create() { return StreamEvent(); }

  StreamEvent();
  ~StreamEvent();

  std::uint64_t
  Id() const;

  // {
  StreamEvent(const StreamEvent&) = delete;
  StreamEvent&
  operator=(const StreamEvent&) = delete;
  // }

  // {
  StreamEvent(StreamEvent&&);
  StreamEvent&
  operator=(StreamEvent&&);
  // }

  ///
  /// Return true if the current event has completed, may throw an `cm::Error`
  /// if an error has occurred on the underlying event.
  ///
  bool
  IsComplete() const;

 private:
  cudaEvent_t event;
  std::uint64_t id;
};
} // ns cmm

#endif // CMM_EVENT_HH_
