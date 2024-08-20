#pragma once
#include "Basics.h"
#include <Eigen/Dense>

namespace Q_Algorithms {

    using Complex = std::complex<double>;
    using VectorXcd = Eigen::VectorXcd;
    using MatrixXcd = Eigen::MatrixXcd;

    // General Kronecker product function for vectors
    VectorXcd kroneckerProduct(const VectorXcd& a, const VectorXcd& b) {
        VectorXcd result(a.size() * b.size());

        for (int i = 0; i < a.size(); ++i) {
            for (int j = 0; j < b.size(); ++j) {
                result(i * b.size() + j) = a(i) * b(j);
            }
        }

        return result;
    }

    // Quantum Phase Estimation function
    template<size_t n, size_t m>
    Qubits<m> QPE(const Eigen::Matrix<Complex, (1 << n), 1>& input, const Eigen::Matrix<Complex, (1 << n), (1 << n)>& U) {
        constexpr size_t Dn = 1 << n; // Dimension for the input state (2^n)
        constexpr size_t Dm = 1 << m; // Dimension for the ancillary qubits (2^m)

        // Step 1: Initialize the ancillary qubits in the |0>^m state
        Eigen::Matrix<Complex, Dm, 1> m_state = Eigen::Matrix<Complex, Dm, 1>::Zero();
        m_state(0) = 1;

        // Step 2: Tensor product of ancillary qubits with the input state
        VectorXcd combined_state = kroneckerProduct(m_state, input);

        // Step 3: Initialize the combined quantum state
        Qubits<n + m> psi(combined_state, MatrixXcd::Identity(Dm * Dn, Dm * Dn));

        // Step 4: Apply Hadamard gates to the ancillary qubits
        for (size_t i = n + m - 1; i >= n; --i) {
            psi.applyGate(Q_Gates::H, i);
        }

        // Step 5: Apply controlled-U gates for phase estimation
        MatrixXcd controlled_U = U;
        for (size_t j = 0; j < m; ++j) {
            // Apply controlled-U^(2^j)
            psi.applyCGate(controlled_U, 0, n - 1, j + n);

            // Update controlled_U to U^(2^(j+1))
            controlled_U *= controlled_U;
        }

        // Further steps involve performing the Inverse Quantum Fourier Transform (IQFT)
        psi.applyIQFT();
        // At this point, the result of the phase estimation will be stored in the first `m` qubits of `psi`.
        Vector<Complex, Dm> result_state = ;
        return Qubits<m>(result_state, MatrixXcd::Identity(Dm, Dm));
    }

}
