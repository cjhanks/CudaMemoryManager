#ifndef CMM_TOOLS_SCHEDULE_HH_
#define CMM_TOOLS_SCHEDULE_HH_

#include <cstring>
#include <cuda_runtime.h>

namespace cmm {
class Schedule1D {
 public:
  static Schedule1D
  MaxThreads(std::size_t work_items);

  static Schedule1D
  MinThreads(std::size_t work_items);

  std::size_t
  WorkItems() const;

  std::size_t
  B() const;

  std::size_t
  T() const;

 private:
  std::size_t work_items;
  std::size_t blocks;
  std::size_t threads;

  Schedule1D(std::size_t work_items, std::size_t blocks, std::size_t threads);
};
} // ns cmm

#endif // CMM_TOOLS_SCHEDULE_HH_
