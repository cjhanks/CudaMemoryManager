
using SampleVector = Vec<complex<double>>;

class ToneGenerator {
 public:
  virtual SampleVector
  Next() = 0;

  // Tone generators provide an infinitely long tone, but if we integrate this
  // with audio parsers, it might be nice if the code was already instrumented
  // with a begin/end or... to indicate error states.
  virtual bool
  More() = 0;
};

///
/// A constant unchanging tone.
///
class ConstantTone : public ToneGenerator {
 public:
  struct Type1 {
    double frequency; // Hz
    double amplitude; // Coefficient scalar

    /// seconds (this should be very small, less than 1/frequency)
    double delay;
  };

  static ToneGenerator*
  Create(DigitalSpecification digispec, Type1 tone);

 private:
};

///
/// Useful for testing the quantization of the audio signal.
///
/// When frequency0 = frequency1, this generates a constant tone.
/// When not equal, a triangle frequency sweep oscillates between the two
/// frequencies.
///
/// The rate of change has two orders:
/// frequency_d0 - (linear 0th order)
/// frequency_d1 - (1st order derivative)
///
class FrequencySweep : public ToneGenerator {
 public:
  struct Type1 {
    /// Frequency start Hz
    double frequency0;

    /// Frequency end Hz
    double frequency1;

    /// 0th, 1st derivative of sweep;
    double frequency_d0;
    double frequency_d1;

    /// Coefficient scalar
    double amplitude;

    /// seconds (this should be very small, less than 1/frequency)
    double delay;
  };

  static ToneGenerator*
  Create(DigitalSpecification digispec, Type1 tone);
};
