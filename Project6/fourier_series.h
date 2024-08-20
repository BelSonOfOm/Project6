#ifndef FOURIER_SERIES_H
#define FOURIER_SERIES_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <memory>
#include "complex.h"

namespace fourier_stuff {
    static constexpr double PI = 3.14159265358979323846;

    using Complex = complex_numbers::Complex;

    class FourierSeries {
    public:
        constexpr FourierSeries(int num_steps, int num_sum)
            : NUM_STEPS{ num_steps }, NUM_SUM{ num_sum } {}

         inline Complex compute_c0(Complex(*f)(double), double a, double b) const {
            return (1.0 / (b - a)) * trapezoidal_rule(f, a, b);
        }

        inline Complex compute_cn(Complex(*f)(double), double a, double b, int n) const {
            return (2.0 / (b - a)) * trapezoidal_rule([=](double x) { return f(x) * complex_exp(Complex(0, n * x)); }, a, b);
        }

        inline Complex compute_sum(Complex(*f)(double), double a, double b) const {
            Complex sum = compute_c0(f, a, b);
            for (int k = 1; k <= NUM_SUM; ++k) {
                sum = sum + compute_cn(f, a, b, k);
            }
            return sum;
        }

    private:
        const int NUM_STEPS;
        const int NUM_SUM;

        template <typename Func>
        constexpr Complex trapezoidal_rule(Func func, double a, double b) const {
            double h = (b - a) / NUM_STEPS;
            Complex sum = 0.5 * (func(a) + func(b));
            for (int i = 1; i < NUM_STEPS; ++i) {
                sum = sum + func(a + i * h);
            }
            return sum * h;
        }

        Complex complex_exp(const Complex& z) const {
            double exp_real = std::exp(z.real());
            double cos_imag, sin_imag;

            // Inline assembly to compute cos and sin
            #if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
            __asm__(
                "fsincos"
                : "=t" (cos_imag), "=u" (sin_imag)
                : "0" (z.imag())
            );
            #else
            cos_imag = std::cos(z.imag());
            sin_imag = std::sin(z.imag());
            #endif

            return Complex(exp_real * cos_imag, exp_real * sin_imag);
        }
    };

    template <typename T, std::size_t N>
    constexpr std::array<Complex, N> to_complex(const std::array<T, N>& real_values) {
        std::array<Complex, N> complex_values;
        for (std::size_t i = 0; i < N; ++i) {
            complex_values[i] = Complex(real_values[i], 0.0);
        }
        return complex_values;
    }



    template <std::size_t N>
    constexpr std::array<Complex, N / 2> compute_roots(double inverse_factor) {
        std::array<Complex, N / 2> roots = {};
        for (std::size_t i = 0; i < N / 2; ++i) {
            double angle = 2 * PI * i / N * inverse_factor;
            roots[i] = Complex(std::cos(angle), std::sin(angle));
        }
        return roots;
    }


    template <typename Container, std::size_t N>
    void fft(Container& a, bool invert) {
        static_assert(N > 0, "Container size must be greater than zero");
        const double inverse_factor = invert ? -1.0 : 1.0;

        constexpr std::array<Complex, N / 2> roots = compute_roots<N>(inverse_factor);

        // Bit reversal permutation
        for (std::size_t i = 1, j = 0; i < N; ++i) {
            std::size_t bit = N >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;

            if (i < j)
                std::swap(a[i], a[j]);
        }

        // Danielson-Lanczos part
        for (std::size_t len = 2; len <= N; len <<= 1) {
            for (std::size_t i = 0; i < N; i += len) {
                for (std::size_t j = 0; j < len / 2; ++j) {
                    Complex u = a[i + j];
                    Complex v = a[i + j + len / 2] * roots[j * (N / len)];
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                }
            }
        }

        // Inverse FFT normalization
        if (invert) {
            double inv_N = 1.0 / static_cast<double>(N);
            for (auto& x : a) {
                x *= inv_N;
            }
        }
    }

}  // namespace fourier_stuff
#endif  // FOURIER_SERIES_H


