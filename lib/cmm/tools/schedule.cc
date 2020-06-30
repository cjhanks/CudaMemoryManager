#include "schedule.hh"

#include <cmath>
#include "cmm/logging.hh"

namespace cmm {
Schedule1D
Schedule1D::MaxThreads(std::size_t work_items)
{
  static constexpr std::size_t MaximumThreads = 1024;
  return Schedule1D(work_items,
                  std::ceil(work_items / double(MaximumThreads)),
                  MaximumThreads);

}

Schedule1D
Schedule1D::MinThreads(std::size_t work_items)
{
  LOG(FATAL) << "TODO";
  static constexpr std::size_t MaximumThreads = 1024;
  std::size_t shift = 0;
  while (work_items >> shift > MaximumThreads)
    shift++;
}

Schedule1D::Schedule1D(
    std::size_t work_items, std::size_t blocks, std::size_t threads)
  : work_items(work_items),
    blocks(blocks),
    threads(threads)
{}

std::size_t
Schedule1D::B() const
{ return blocks; }

std::size_t
Schedule1D::T() const
{ return threads; }
}
