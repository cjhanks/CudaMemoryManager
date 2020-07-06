#ifndef CMM_DIGITAL_HH_
#define CMM_DIGITAL_HH_

namespace cmm {
///
/// @class DigitalSpecification
///
/// A specification for the digital sampling characteristics.
///
class DigitalSpecification {
 public:
  ///
  /// Number of audio channels in signal.
  ///
  std::size_t
  Channels() const;

  ///
  /// Number of samples in a second.
  ///
  std::size_t
  SampleRate() const;

 private:
  // Samples per second.
  std::size_t samples_rate;

  // Number of audio channels.
  std::size_t channels;
};
} // ns cmm

#endif // CMM_DIGITAL_HH_
