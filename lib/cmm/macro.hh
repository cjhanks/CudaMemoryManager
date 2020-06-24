#ifndef CMM_MACRO_HH_
#define CMM_MACRO_HH_

#define cmm_likely(x)       __builtin_expect(!!(x), 1)
#define cmm_unlikely(x)     __builtin_expect(!!(x), 0)

#endif // CMM_MACRO_HH_
