#ifndef CMM_FFT_HH_
#define CMM_FFT_HH_

#include "cmm/matrix.hh"

namespac cmm {
///
/// @class FFT_1D
///
/// This class performs a batched 1D FFT on a 2D matrix or a single FFT on a 1D
/// vector.
///
/// Typically this class would be used for slow-time quantization of the
/// continuous stream, and will be necessary for frequency based modulations.
///
/// [0] -> SlowTime
/// [1] -> FastTime
///
template <typename Type>
class FFT_1D {
 public:
  struct Options {
    /// Perform IFFT
    bool inverse = false;

    /// CUFFT is a non-normalized FFT, this can add undesired gain.  So it
    /// should be multipled by the inverse of the fast-time.
    bool scaled  = true;
  };

  explicit FFT_1D(Options options)
    : options(options),
      initialized(false)
  {
  }

  template <typename Memory>
  void
  Apply(Matrix<Type, 1, Memory>& data)
  {
    using FFT = bit::FFT<Type>;
    if (options.scaled)
      data *= FFT::Float(data.Size(0));
  }

  template <typename Memory>
  void
  Apply(Matrix<Type, 2, Memory>& data)
  {
    using FFT = bit::FFT<Type>;
    if (options.scaled)
      data *= FFT::Float(data.Size(0));
  }

 private:
  Options options;
  bool initialized;
};

///
/// @class FFT_2D
///
/// This class performs a single 2D FFT on a 2D matrix.
/// This class will be an experiment to identify
///
class FFT_2D {
 public:
  template <typename Memory>
  void
  Apply(Matrix<Type, 2, Memory>& data)
  {
  }
};
} // ns cmm

#endif // CMM_FFT_HH_
