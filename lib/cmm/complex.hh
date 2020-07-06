#ifndef CMM_COMPLEX_HH_
#define CMM_COMPLEX_HH_

#include <complex>

#include "cmm/macro.hh"
#include "complex-bit.hh"


namespace cmm {
template <typename Type>
class Complex {
 public:
  using Cuda = typename bit::Complex<Type>;
  using Self = Complex<Type>;

  cmm_method
  Complex() = default;

  cmm_method
  Complex(Type real, Type imag)
    : data(Cuda::make(real, imag)) {}

  cmm_method
  Complex(typename Cuda::Type data)
    : data(data) {}

  Complex(const std::complex<Type>& rhs)
    : Complex(rhs.real(), rhs.imag()) {}

  Complex&
  operator=(std::complex<Type>& rhs)
  {
    data.x = rhs.real();
    data.y = rhs.imag();
    return *this;
  }

  bool
  operator==(const Complex& rhs) const
  { return real() == rhs.real()
        && imag() == rhs.imag(); }

  bool
  operator==(const std::complex<Type>& rhs) const
  { return real() == rhs.real()
        && imag() == rhs.imag(); }

  /// {
  /// Convenience casting
  cmm_method
  operator typename Cuda::Type() const
  { return data; }

  operator std::complex<Type>() const
  { return std::complex<Type>(real(), imag()); }

  template <typename NewType>
  Complex<NewType>
  Cast()
  { return Complex<NewType>(real(), imag()); }
  /// }

  /// {
  /// Accessors.
  cmm_method
  Type&
  real()
  { return data.x; }

  cmm_method
  const Type&
  real() const
  { return data.x; }

  cmm_method
  Type&
  imag()
  { return data.y; }

  cmm_method
  const Type&
  imag() const
  { return data.y; }
  /// }

  cmm_method
  Self&
  operator+=(const Self& rhs)
  {
    data = Cuda::add(data, rhs.data);
    return *this;
  }

  cmm_method
  Self&
  operator-=(const Self& rhs)
  {
    data = Cuda::sub(data, rhs.data);
    return *this;
  }

  cmm_method
  Self&
  operator*=(const Self& rhs)
  {
    data = Cuda::mul(data, rhs.data);
    return *this;
  }

  cmm_method
  Self&
  operator/=(const Self& rhs)
  {
    data = Cuda::div(data, rhs.data);
    return *this;
  }

  cmm_method
  Self
  operator+(const Self& rhs) const
  { return Cuda::add(data, rhs.data); }

  cmm_method
  Self
  operator-(const Self& rhs) const
  { return Cuda::sub(data, rhs.data); }

  cmm_method
  Self
  operator*(const Self& rhs) const
  { return Cuda::mul(data, rhs.data); }

  cmm_method
  Self
  operator*(const Type& rhs) const
  { return Self(real() * rhs,
                imag() * rhs); }

  cmm_method
  Self
  operator/(const Self& rhs) const
  { return Cuda::div(data, rhs.data); }

  cmm_method
  Self
  operator/(const Type& rhs) const
  { return Self(real() / rhs,
                imag() / rhs); }

 private:
  typename Cuda::Type data;
};

template <typename Type>
cmm_method
Type
real(const Complex<Type>& data) { return data.real(); }

template <typename Type>
cmm_method
Type
imag(const Complex<Type>& data) { return data.imag(); }

template <typename Type>
cmm_method
Type
abs(const Complex<Type>& data)
{
  using Cuda = typename Complex<Type>::Cuda;
  return Cuda::abs(data);
}

template <typename Type>
cmm_method
Type
arg(const Complex<Type>& data)
{
  return atan2(data.imag(), data.real());
}

template <typename Type>
cmm_method
Type
norm(const Complex<Type>& data)
{
  return pow(data.real(), Type(2.0))
       + pow(data.imag(), Type(2.0));
}

template <typename Type>
cmm_method
Complex<Type>
conj(const Complex<Type>& data)
{
  using Cuda = typename Complex<Type>::Cuda;
  return Complex<Type>(Cuda::conj(data));
}

template <typename Type>
cmm_method
Complex<Type>
proj(const Complex<Type>& data)
{
  auto d = norm(data) + Type(1);
  return Complex<Type>(Type(2) * data.real() / d,
                       Type(2) * data.imag() / d);
}

template <typename Type>
cmm_method
Complex<Type>
polar(const Type& r, const Type& theta = Type(0))
{ return Complex<Type>(r * cos(theta), r * sin(theta)); }

template <typename Type>
cmm_method
Complex<Type>
exp(const Complex<Type>& data)
{
  return Complex<Type>(data.real() * cos(data.imag()),
                       data.real() * sin(data.imag()));
}

template <typename Type>
cmm_method
Complex<Type>
log(const Complex<Type>& data)
{ return Complex<Type>(std::log(abs(data)), arg(data)); }

template <typename Type>
cmm_method
Complex<Type>
logN(const Complex<Type>& data, Type n)
{ return log(data) / std::log(Type(n)); }

template <typename Type>
cmm_method
Complex<Type>
log10(const Complex<Type>& data)
{ return LogN(data, Type(10)); }
} // ns cmm

#endif // CMM_COMPLEX_HH_
