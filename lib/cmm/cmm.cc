#include "cmm.hh"

#include "cmm/memory-bit.hh"

namespace cmm {
void
Install(Specification specification)
{
  bit::InstallDiscretizer(std::move(specification.discretizer));
}
} // ns cmm
