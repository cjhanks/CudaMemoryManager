
#include "cmm/cmm.hh"

int
main()
{
  cmm::Specification cmm_spec;
  cmm_spec.discretizer.reset(new cmm::DiscretizerRounding(32));
  cmm::Install(std::move(cmm_spec));


}
