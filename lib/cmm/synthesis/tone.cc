#include "tone.hh"

namespace cmm {
ToneGenerator*
ConstantTone::Create(DigitalSpecification digispec, Type1 tone)
{
  double sample_spacing = 1.0 / digispec.SampleRate();
  double period = 1.0 / tone.frequency;
  std::size_t samples = period / sample_spacing;
}
} // ns cmm
