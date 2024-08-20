#include <array>
#include <emmintrin.h> // SSE2 intrinsics
#include <pmmintrin.h> // SSE3 intrinsics
#include <tmmintrin.h> // SSSE3 intrinsics
#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace complex_numbers{

    class Complex {
    public:
        Complex() : data{ _mm_setzero_pd() } {}
        Complex(double r, double i) : data{ _mm_set_pd(i, r) } {} 
        explicit Complex(const std::complex<double>& other) : data{ _mm_set_pd(other.imag(), other.real()) } {}

        inline double real() const {
            double result[2];
            _mm_store_pd(result, data);
            return result[0];
        }

        inline double imag() const {
            double result[2];
            _mm_store_pd(result, data);
            return result[1];
        }

        inline void set_real(double r) {
            data = _mm_move_sd(_mm_set_sd(r), data);
        }

        inline void set_img(double i) {
            data = _mm_move_sd(_mm_set_sd(i), _mm_unpackhi_pd(data, data));
        }

        inline double abs() const {
            __m128d squared = _mm_mul_pd(data, data);
            double result[2];
            _mm_store_pd(result, squared);
            return std::sqrt(result[0] + result[1]);
        }

        inline Complex conj() const {
            double result[2];
            _mm_store_pd(result, data);
            return Complex(result[0], -result[1]);
        }

        inline Complex operator+(const Complex& other) const {
            return Complex(_mm_add_pd(data, other.data));
        }

        inline Complex operator-(const Complex& other) const {
            return Complex(_mm_sub_pd(data, other.data));
        }

        inline Complex operator*(const Complex& other) const {
            __m128d a = data;
            __m128d b = other.data;
            __m128d ac_bd = _mm_mul_pd(a, _mm_shuffle_pd(b, b, 0x0)); // (a*c, b*d)
            __m128d ad_bc = _mm_mul_pd(a, _mm_shuffle_pd(b, b, 0x3)); // (a*d, b*c)
            __m128d real_img = _mm_addsub_pd(ac_bd, _mm_shuffle_pd(ad_bc, ad_bc, 0x1)); // (a*c - b*d, a*d + b*c)
            return Complex(real_img);
        }

        inline Complex operator/(const Complex& other) const {
            __m128d a = data;
            __m128d b = other.data;
            __m128d ac_bd = _mm_mul_pd(a, _mm_shuffle_pd(b, b, 0x0)); // (a*c, b*d)
            __m128d ad_bc = _mm_mul_pd(a, _mm_shuffle_pd(b, b, 0x3)); // (a*d, b*c)
            __m128d real_img = _mm_addsub_pd(ac_bd, _mm_shuffle_pd(ad_bc, ad_bc, 0x1)); // (a*c + b*d, b*c - a*d)
            __m128d bb_aa = _mm_mul_pd(b, b); // (c*c, d*d)
            __m128d denom = _mm_hadd_pd(bb_aa, bb_aa); // (c*c + d*d, c*c + d*d)
            return Complex(_mm_div_pd(real_img, denom));
        }
        inline bool operator ==(const Complex& other) const {
            double result[2];
            _mm_store_pd(result, data);
            return ( result[0] == other.real() && result[1] == other.imag());
        }

        inline bool operator !=(const Complex& other) const {
            return !(*(this) == other);
        }

        inline Complex operator+(double val) const {
            __m128d v = _mm_set_sd(val);
            __m128d result = _mm_add_sd(data, v);
            return Complex(result);
        }

        inline Complex operator-(double val) const {
            __m128d v = _mm_set_sd(val);
            __m128d result = _mm_sub_sd(data, v);
            return Complex(result);
        }

        inline Complex operator*(double val) const {
            __m128d v = _mm_set1_pd(val);
            return Complex(_mm_mul_pd(data, v));
        }

        inline Complex operator/(double val) const {
            __m128d v = _mm_set1_pd(val);
            return Complex(_mm_div_pd(data, v));
        }

        inline Complex& operator=(const Complex& other) {
            if (this != &other) {
                data = other.data;
            }
            return *this;
        }

        inline Complex& operator+=(const Complex& other) {
            data = _mm_add_pd(data, other.data);
            return *this;
        }

        inline Complex& operator*=(const Complex& other) {
            *this = *this * other;
            return *this;
        }

        inline Complex& operator*=(double val) {
            data = _mm_mul_pd(data, _mm_set1_pd(val));
            return *this;
        }

        inline Complex& operator=(double val) {
            data = _mm_set_sd(val);
            return *this;
        }

        inline Complex& operator=(const std::complex<double>& other) {
            data = _mm_set_pd(other.imag(), other.real());
            return *this;
        }

        inline double theta() const { return std::atan2(imag(), real()); }

        inline Complex exp() const {
            double exp_real = std::exp(real());
            return Complex(exp_real * std::cos(imag()), exp_real * std::sin(imag()));
        }

        inline operator std::complex<double>() const {
            double result[2];
            _mm_store_pd(result, data);
            return std::complex<double>(result[0], result[1]);
        }

        inline friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
            double result[2];
            _mm_store_pd(result, c.data);
            os << "(" << result[0] << ", " << result[1] << ")";
            return os;
        }

        inline friend Complex operator+(double lhs, const Complex& rhs) {
            return rhs + lhs;
        }

        inline friend Complex operator-(double lhs, const Complex& rhs) {
            return Complex(lhs - rhs.real(), -rhs.imag());
        }

        inline friend Complex operator*(double lhs, const Complex& rhs) {
            return rhs * lhs;
        }

        inline friend Complex operator/(double lhs, const Complex& rhs) {
            double denom = rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
            return Complex(lhs * rhs.real() / denom, -lhs * rhs.imag() / denom);
        }

    private:
        __m128d data;

        Complex(__m128d d) : data(d) {}
    };

} // namespace complex_numbers

namespace Eigen {
    template<>
    struct NumTraits<complex_numbers::Complex> : NumTraits<double> {
        typedef complex_numbers::Complex Real;
        typedef complex_numbers::Complex NonInteger;
        typedef complex_numbers::Complex Nested;
        enum {
            IsComplex = 1,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 2,
            AddCost = 4,
            MulCost = 8
        };

        static inline Real epsilon() { return Real(std::numeric_limits<double>::epsilon(), 0); }
        static inline Real dummy_precision() { return Real(std::numeric_limits<double>::epsilon() * 1000, 0); }
    };

    template<>
    struct ScalarBinaryOpTraits<complex_numbers::Complex, complex_numbers::Complex, internal::scalar_sum_op<complex_numbers::Complex>> {
        typedef complex_numbers::Complex ReturnType;
    };

    template<>
    struct ScalarBinaryOpTraits<complex_numbers::Complex, complex_numbers::Complex, internal::scalar_difference_op<complex_numbers::Complex>> {
        typedef complex_numbers::Complex ReturnType;
    };

    template<>
    struct ScalarBinaryOpTraits<complex_numbers::Complex, complex_numbers::Complex, internal::scalar_product_op<complex_numbers::Complex>> {
        typedef complex_numbers::Complex ReturnType;
    };

    template<>
    struct ScalarBinaryOpTraits<complex_numbers::Complex, complex_numbers::Complex, internal::scalar_quotient_op<complex_numbers::Complex>> {
        typedef complex_numbers::Complex ReturnType;
    };
}