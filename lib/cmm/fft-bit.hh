#ifndef Self_FFT_BIT_HH_
#define Self_FFT_BIT_HH_

#include "cmm/complex.hh"

namespace cmm { namespace bit {

template <typename Type>
struct FFT;;

template <>
struct FFT<Complex<double>> {
  using CuType = typename bit::Complex<cmm::Complex<double>>::Type;
  using Float = double;

  static constexpr auto FftForward = CUFFT_Z2Z;
  static constexpr auto FftInverse = CUFFT_Z2Z;
  using InverseType = FFT<Complex<double>>;
};

template <>
struct FFT<Complex<float>> {
  using CuType = typename bit::Complex<cmm::Complex<float>>::Type;
  using Float = float;

  static constexpr auto FftForward = CUFFT_C2C;
  static constexpr auto FftInverse = CUFFT_C2C;
  using InverseType = FFT<Complex<float>>;
};

template <>
struct FFT<double> {
  using CuType = double;
  using Float = double;

  static constexpr auto FftForward = CUFFT_Z2D;
  static constexpr auto FftInverse = CUFFT_D2Z;
  using InverseType = FFT<Complex<double>>;
};

template <>
struct FFT<float> {
  using CuType = float;
  using Float = float;

  static constexpr auto FftForward = CUFFT_R2C;
  static constexpr auto FftInverse = CUFFT_C2R;
  using InverseType = FFT<Complex<float>>;
};
} // ns bit
} // ns cmm
#endif // Self_FFT_BIT_HH_
