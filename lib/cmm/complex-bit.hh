#ifndef CMM_COMPLEX_BIT_HH_
#define CMM_COMPLEX_BIT_HH_

#include <cuda_runtime.h>
#include <cuComplex.h>


namespace cmm { namespace bit {
template <typename Type>
struct Complex;

template <>
struct Complex<double> {
  using Type = cuDoubleComplex;

  static constexpr auto make = make_cuDoubleComplex;
  static constexpr auto real = cuCreal;
  static constexpr auto imag = cuCimag;
  static constexpr auto conj = cuConj;
  static constexpr auto mul  = cuCmul;
  static constexpr auto div  = cuCdiv;
  static constexpr auto add  = cuCadd;
  static constexpr auto sub  = cuCsub;
  static constexpr auto abs  = cuCabs;
};

template <>
struct Complex<float> {
  using Type = cuFloatComplex;

  static constexpr auto make = make_cuFloatComplex;
  static constexpr auto real = cuCrealf;
  static constexpr auto imag = cuCimagf;
  static constexpr auto conj = cuConjf;
  static constexpr auto mul  = cuCmulf;
  static constexpr auto div  = cuCdivf;
  static constexpr auto add  = cuCaddf;
  static constexpr auto sub  = cuCsubf;
  static constexpr auto abs  = cuCabsf;
};
} // ns bit
} // ns cmm

#endif // CMM_COMPLEX_BIT_HH_
