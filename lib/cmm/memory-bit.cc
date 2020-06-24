#include "memory-bit.hh"


namespace cmm { namespace bit {
namespace {
MemoryManager mm;
} // ns

void
InstallDiscretizer(std::unique_ptr<Discretizer>&& discretizer)
{
  mm.Install(std::move(discretizer));
}

void
MemoryManager::Install(std::unique_ptr<Discretizer>&& discretizer)
{
  this->discretizer = std::move(discretizer);
}
} // ns bit
} // ns cmm
