#include "assert.hh"

#include <cstdlib>
#include <iostream>

namespace cmm {
void
Assert(const char* file, unsigned line, const char* condition)
{
  std::cerr << file << ":" << line << " " << condition << std::endl;
  std::exit(-1);
}
} // ns cmm

