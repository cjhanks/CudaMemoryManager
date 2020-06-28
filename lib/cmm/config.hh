#ifndef CMM_CONFIG_HH_
#define CMM_CONFIG_HH_

#include <memory>

namespace cmm {
// Simple forward-declaration.
class Discretizer;

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
  /// Time quanta for the event creator in modulo.
  /// Ie:
  ///   1 - Create an event 100% of the time..
  ///   2 - Create an event  50% of the time
  ///   3 - Create an event  33% of the time
  ///   ...
  ///
  /// Some programs may be generating so much memory traffic that they backlog
  /// the queue.  The result is too many `cmm::Event` classes are created and
  /// the underlying resources are exhausted.
  ///
  /// In this event, it would probably be better for the program to create its
  /// own memory pool.
  ///
  /// But you can also reduce the number of events which are created in the
  /// memory pool by having it only
  ///
  std::size_t time_quanta_ev_modulo = 1;

  ///
  /// Time quanta for the garbage collector in microseconds.
  ///
  /// The memory garbage collector runs in two instances:
  /// - New memory has been freed.
  /// - `time_quanta_gc_us`
  ///
  std::size_t time_quanta_gc_us = 100;

  ///
  /// Specify the Discretizer, see the `cmm::Discretizer` for explanation of
  /// this field.
  ///
  std::unique_ptr<Discretizer> discretizer;
};

void
Install(Specification&& specification);
} // ns cmm

#endif // CMM_CONFIG_HH_
