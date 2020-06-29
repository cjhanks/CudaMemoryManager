#include "scope.hh"

#include <cuda_runtime.h>
#include "cmm/tools/error.hh"
#include "cmm/tools/stream.hh"

namespace cmm {
SyncScope::~SyncScope()
{
  Stream::This().Synchronize();
}

Canary::Canary()
{
  Error::Check(cudaPeekAtLastError());
}

Canary::~Canary()
{
  Error::Check(cudaPeekAtLastError());
}

void
Canary::Check()
{
  Error::Check(cudaPeekAtLastError());
}
} // ns cmm
