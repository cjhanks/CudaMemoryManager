#ifndef CMM_STREAM_HH_
#define CMM_STREAM_HH_

#include <cuda_runtime.h>

#include "cmm.hh"

namespace cmm {
///
/// @class Stream
///
/// This class encapsulates the lifetime of a cudaStream_t.  When
/// cuda_api_per_thread_default_stream is set, this class functionally acts as a
/// noop that returns the default stream.
///
class Stream {
 public:
  ///
  /// Get the thread local instance of the `Stream`.
  ///
  static Stream&
  This();

  Stream();
  ~Stream();

  /// {
  /// Helpful implicit casting.
  operator cudaStream_t();
  operator cudaStream_t() const;
  /// }

 private:
  cudaStream_t stream;
};

namespace bit {
void
InstallCudaStream(Specification& spec);
} // ns bit
} // ns cmm

#endif // CMM_STREAM_HH_
