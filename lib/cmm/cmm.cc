#include "cmm.hh"

#include "cmm/memory-bit.hh"
#include "cmm/stream.hh"

namespace cmm {
void
Install(Specification specification)
{
  bit::InstallDiscretizer(std::move(specification.discretizer));
  bit::InstallCudaStream(specification);
}
} // ns cmm
