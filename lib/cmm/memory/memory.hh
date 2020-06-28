#ifndef CMM_MEMORY_HH_
#define CMM_MEMORY_HH_

#include <cstring>
#include <memory>

namespace cmm {
///
/// @class Discretizer
///
/// The memory allocator will not perform well if there are many different size
/// classes which need to be tracked.  Different applications have different
/// allocation patterns, there exists no universally correct pattern.
///
/// This class allows you to implement different rounding strategies to force
/// the requested bytes to round up to a discrete subset of all integer values.
///
class Discretizer {
 public:
  ///
  /// Receive the bytes requested and return the number of bytes the allocator
  /// should allocate.  This must be greater than or equal to `bytes`.
  ///
  virtual std::size_t
  Compute(std::size_t bytes) const = 0;
};

///
/// @class DiscretizerNone
///
/// Do not discretize, simply return the number of bytes requested.
///
class DiscretizerNone : public Discretizer {
 public:
  virtual std::size_t
  Compute(std::size_t bytes) const;
};

///
/// @class DiscretizerRounding
///
/// Discretize by rounding up to the nearest `rounded_value` specified in the
/// constructor.
///
class DiscretizerRounding : public Discretizer {
 public:
  explicit DiscretizerRounding(std::size_t rounded_value);

  virtual std::size_t
  Compute(std::size_t bytes) const;

 private:
  std::size_t rounded_value;
};
} // ns cmm

#endif // CMM_MEMORY_HH_
