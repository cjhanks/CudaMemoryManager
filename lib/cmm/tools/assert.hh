#ifndef CMM_ASSERT_HH_
#define CMM_ASSERT_HH_

#include <functional>

#define CMM_ASSERT(_condition_)                       \
  do {                                                \
    if ( ! (_condition_) )                            \
      cmm::Assert(__FILE__, __LINE__, #_condition_); \
  } while (0)

namespace cmm {
void
Assert(const char* file, unsigned line, const char* condition);
} // ns cmm

#endif // CMM_CMM_HH_
