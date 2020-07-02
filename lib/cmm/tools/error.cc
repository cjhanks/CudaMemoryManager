#include "error.hh"

namespace cmm {
Error::Error(std::string message)
  : std::runtime_error(message) {}

void
Error::Throw(cudaError_t rc)
{
  // Reset the error.
  // FIXME:  When used in conjunction with the Canary class, there is
  //         technically a race condition in multi-threaded applications, though
  //         there is no clear fix for this.
  cudaGetLastError();

  // Special cases which should not throw.
  if (cudaErrorCudartUnloading == rc)
    return;


  throw Error(std::string(cudaGetErrorName(rc))
            + " [" + std::to_string(rc) + "]");
}
} // ns cmm
