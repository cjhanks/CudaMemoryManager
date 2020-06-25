#ifndef CMM_CMM_HH_
#define CMM_CMM_HH_

#include <memory>

#include "cmm/memory.hh"

namespace cmm {
///
/// @struct Specification
///
/// This structure allows for all of the the cmm library options to be
/// configured in a single place.
///
/// All options here must be set before any other cmm calls are made.
///
struct Specification {
  ///
  /// Since CUDA7, NVCC has allowed compilation to make the default CUDA stream
  /// thread local.  However, not all people implemenent this command flag.
  ///
  /// In the event that it is not enabled, a thread_local stream will be
  /// assigned to every thread that can be accessed via the `cmm::Stream` class.
  ///
  bool cuda_api_per_thread_default_stream = false;

  ///
  /// Specify the Discretizer, see the `cmm::Discretizer` for explanation of
  /// this field.
  ///
  std::unique_ptr<Discretizer> discretizer;
};

void
Install(Specification&& specification);
} // ns cmm

#endif // CMM_CMM_HH_
