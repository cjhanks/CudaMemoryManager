#include "error.hh"

namespace cmm {
Error::Error(std::string message)
  : std::runtime_error(message) {}

void
Error::Throw(cudaError_t rc)
{
  // Special cases which should not throw.
  if (cudaErrorCudartUnloading == rc)
    return;

  throw Error(std::string(cudaGetErrorName(rc))
            + " [" + std::to_string(rc) + "]");
}

// -------------------------------------------------------------------------- //

Canary::~Canary()
{
  Error::Check(cudaGetLastError());
}

} // ns cmm
