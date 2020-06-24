#ifndef CMM_ERROR_HH_
#define CMM_ERROR_HH_

#include <stdexcept>
#include <cuda_runtime.h>

#include "macro.hh"

namespace cmm {
///
/// @class Error
///
class Error : public std::runtime_error {
 public:
  static void
  Check(cudaError_t rc) {
    if (cmm_unlikely(rc != 0)) {
      Throw(rc);
    }
  }

  explicit Error(std::string message);

 private:
  static void
  Throw(cudaError_t rc);
};
} // ns cmm

#endif // CMM_ERROR_HH_
