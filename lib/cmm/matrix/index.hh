#ifndef CMM_MATRIX_INDEX_HH_
#define CMM_MATRIX_INDEX_HH_

#include <cstdint>
#include "cmm/logging.hh"


namespace cmm {
///
/// Acts as an ND->1D mapping function.  It is constructed with dimensions, and
/// then you can query its sizes and compute the offset.
///
template <std::size_t Dims_>
class Indexer {
 public:
  static constexpr std::size_t Dims = Dims_;

  Indexer()
    : sizes {0},
      jumps {0},
      size(0)
  {}

  Indexer(const Indexer& rhs)
    : sizes {rhs.sizes},
      jumps {rhs.jumps},
      size(rhs.size)
  {
  }

  template <typename... Args>
  Indexer(Args... args)
    : Indexer()
  {
    static_assert(sizeof...(args) == Dims, "Invalid number of args in constructor");

    Initialize(0, args...);

    std::size_t jump = 1;
    for (ssize_t n = Dims - 1; n >= 0; n--) {
      jumps[n] = jump;
      jump *= sizes[n];
    }

    size = jump;
  }

  std::size_t
  Size() const
  { return size; }

  std::size_t
  Size(std::size_t index) const
  { return sizes[index]; }

  template <typename... Args>
  std::size_t
  Index(Args... args)
  {
    return ImplIndex(0, args...);
  }

 private:
  std::uint32_t sizes[Dims];
  std::uint32_t jumps[Dims];
  std::size_t   size;

  template <typename... Args>
  void
  Initialize(std::size_t index, std::size_t value, Args... args)
  {
    sizes[index] = value;
    Initialize(index + 1, args...);
  }

  void
  Initialize(std::size_t index)
  { (void) index; }

  template <typename... Args>
  std::size_t
  ImplIndex(std::size_t index, std::size_t offset, Args... args) const
  {
    return jumps[index] * offset + ImplIndex(index + 1, args...);
  }

  std::size_t
  ImplIndex(std::size_t index) const
  {
    (void) index;
    return 0;
  }
};
} // ns cmm

#endif // CMM_MATRIX_INDEX_HH_
