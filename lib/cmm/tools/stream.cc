#include "stream.hh"

#include "cmm/config.hh"
#include "cmm/tools/error.hh"

namespace cmm {
namespace bit {
namespace {
bool cuda_api_per_thread_default_stream = false;
} // ns

void
InstallCudaStream(Specification& spec)
{
  cuda_api_per_thread_default_stream = spec.cuda_api_per_thread_default_stream;
}
} // ns bit

Stream&
Stream::This()
{
  thread_local Stream stream;
  return stream;
}

Stream::Stream()
{
  if (bit::cuda_api_per_thread_default_stream)
    stream = nullptr;
  else
    Error::Check(cudaStreamCreate(&stream));
}

Stream::~Stream()
{
  if (stream != nullptr)
    Error::Check(cudaStreamDestroy(stream));
}

Stream::operator cudaStream_t()
{ return stream; }

Stream::operator cudaStream_t() const
{ return stream; }
} // ns cmm
