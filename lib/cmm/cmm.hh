#ifndef CMM_CMM_HH_
#define CMM_CMM_HH_

#include <memory>

#include "cmm/memory.hh"

namespace cmm {
struct Specification {
  std::unique_ptr<Discretizer> discretizer;
};

void
Install(Specification&& specification);
} // ns cmm

#endif // CMM_CMM_HH_
