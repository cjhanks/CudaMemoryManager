#include "config.hh"

#include "cmm/memory.hh"
#include "cmm/tools/stream.hh"

namespace cmm {
void
Install(Specification&& specification)
{
  bit::InstallDiscretizer(std::move(specification.discretizer));
  bit::InstallCudaStream(specification);
}
} // ns cmm
