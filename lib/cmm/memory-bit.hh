#ifndef CMM_MEMORY_BIT_HH_
#define CMM_MEMORY_BIT_HH_

#include <memory>
#include <thread>

#include "cmm/memory.hh"


namespace cmm { namespace bit {
///
/// Internal function to install which discretizer should be used globally.
///
void
InstallDiscretizer(std::unique_ptr<Discretizer>&& discretizer);

///
/// @class MemoryManager
///
class MemoryManager {
 public:
  MemoryManager() = default;

  void
  Install(std::unique_ptr<Discretizer>&& discretizer);

 private:
  std::unique_ptr<Discretizer> discretizer;
  std::thread thread_handle;

  /// Launches the thread, this should be called by Install(...)
  void
  Start();
};
} // ns bit
} // ns cmm

#endif // CMM_MEMORY_BIT_HH_
