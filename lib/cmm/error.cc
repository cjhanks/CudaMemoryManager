#include "error.hh"

namespace cmm {
Error::Error(std::string message)
  : std::runtime_error(message) {}

void
Error::Throw(cudaError_t rc)
{
  throw Error(std::string(cudaGetErrorName(rc))
            + " [" + std::to_string(rc) + "]");
}

// -------------------------------------------------------------------------- //

Canary::~Canary()
{
  Error::Check(cudaGetLastError());
}

} // ns cmm
