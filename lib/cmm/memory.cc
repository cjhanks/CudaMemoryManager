#include "memory.hh"

namespace cmm {
// -

std::size_t
DiscretizerNone::Compute(std::size_t bytes) const
{
  return bytes;
}

// -

DiscretizerRounding::DiscretizerRounding(
    std::size_t rounded_value)
  : rounded_value(rounded_value)
{}

std::size_t
DiscretizerRounding::Compute(std::size_t bytes) const
{
  if (bytes % rounded_value)
    return rounded_value * (1 + (bytes / rounded_value));
  else
    return bytes;
}

// -

} // ns cmm
