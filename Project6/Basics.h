#include <iostream>
#include <array>
#include <complex>
#include <Eigen/Dense>

#include<omp.h>

#include <stdexcept>

using namespace Eigen;
using Complex = std::complex<double>;

constexpr auto PI = 3.141592653589793238462643383279;

class QuantumError : public std::runtime_error {
public:
    explicit QuantumError(const std::string& message) : std::runtime_error(message) {}
};

class DimensionMismatchError : public QuantumError {
public:
    DimensionMismatchError() : QuantumError("Dimension mismatch error") {}
};

class InvalidDensityMatrixError : public QuantumError {
public:
    InvalidDensityMatrixError() : QuantumError("Invalid density matrix error") {}
};

namespace q_utils {

    inline bool is_PureState(const Matrix<Complex, 2, 2>& rho) {
        double trace_rho_squared = (rho * rho).trace().real();
        return abs(trace_rho_squared - 1.0) < 1e-10;
    }

    inline bool isBasisStateWithQubit0(int i, size_t qubit_index) {
        return ((i >> qubit_index) & 1) == 0;
    }

    inline bool isBasisStateWithQubit1(int i, size_t qubit_index) {
        return ((i >> qubit_index) & 1) == 1;
    }
}

class Qubit {
public:
    Qubit() : state(Vector<Complex, 2>::Zero()), Bases(Matrix<Complex, 2, 2>::Identity()) {}

    explicit Qubit(const Vector<Complex, 2>& val, const Matrix<Complex, 2, 2>& bases) : Bases(bases) {
        Complex norm = val.norm();
        if (norm != Complex(0, 0)) {
            state = val / norm;
        }
        else {
            state = Vector<Complex, 2>::Zero();
        }
    }

    explicit Qubit(const Matrix<Complex, 2, 2>& rho) {
        if (q_utils::is_PureState(rho)) {
            SelfAdjointEigenSolver<Matrix<Complex, 2, 2>> solver(rho);
            auto eigenvalues = solver.eigenvalues();
            auto eigenvectors = solver.eigenvectors();

            for (int i = 0; i < 2; ++i) {
                if (abs(eigenvalues(i) - 1.0) < 1e-10) {
                    state = eigenvectors.col(i);
                    break;
                }
            }
        }
        else {
            throw InvalidDensityMatrixError();
        }
        Bases = Matrix<Complex, 2, 2>::Identity();
    }

    Qubit(const Qubit& other) : state(other.state), Bases(other.Bases) {}

    inline Matrix<Complex, 2, 2> get_Bases() const { return Bases; }
    inline Vector<Complex, 2> get_state() const { return state; }

    inline Complex operator[](const size_t& i) const { return state[i]; }

    inline void operator>>(const Matrix<Complex, 2, 2>& Bp) {
        Matrix<Complex, 2, 2> Bp_inv = Bp.inverse();
        Bases = Bp;
        state = Bp_inv * state;
    }

    inline Qubit operator+(const Qubit& other) const {
        if (Bases != other.Bases) {
            throw std::invalid_argument("Cannot add qubits with different bases.");
        }
        return Qubit(state + other.state, Bases);
    }

    inline Qubit operator-(const Qubit& other) const {
        if (Bases != other.Bases) {
            throw std::invalid_argument("Cannot subtract qubits with different bases.");
        }
        return Qubit(state - other.state, Bases);
    }

    inline Complex operator*(const Qubit& other) const {
        if (Bases != other.Bases) {
            throw std::invalid_argument("Cannot compute inner product of qubits with different bases.");
        }
        return state.dot(other.state);
    }

    inline void operator*(const Matrix<Complex, 2, 2>& Bp) {
        state = Bp * state;
    }

    inline void operator=(const Qubit& other) {
        state = other.get_state();
        Bases = other.get_Bases();
    }

    inline bool operator==(Qubit& other) const {
        other >> Bases;
        Complex lambda = (*this * other) / std::norm((*this) * other);
        return (lambda == Complex(1, 0));
    }

    inline Matrix<Complex, 2, 2> density_matrix() const {
        return state * state.adjoint();
    }

    void measure_qubit() {
        (*this) >> Matrix<Complex, 2, 2>::Identity();

        double p0 = std::norm(state[0]);  // Probability of measuring 0
        double p1 = std::norm(state[1]);  // Probability of measuring 1

        // Simulate measurement outcome
        double r = static_cast<double>(rand()) / RAND_MAX;
        bool measured_0 = (r < p0);
        // Collapse the qubit based on measurement outcome
        if (measured_0) {
            (*this) = Qubit({ Complex(1, 0), Complex(0, 0) }, Matrix<Complex, 2, 2>::Identity());
        }
        else {
            (*this) = Qubit({ Complex(0, 0), Complex(0, 1) }, Matrix<Complex, 2, 2>::Identity());
        }
    }

private:
    Vector<Complex, 2> state;
    Matrix<Complex, 2, 2> Bases;
};

namespace q_utils {
    inline Qubit create_zero_state() {
        return Qubit(Vector<Complex, 2>({ Complex(1, 0), Complex(0, 0) }), Matrix<Complex, 2, 2>::Identity());
    }

    inline Qubit create_one_state() {
        return Qubit(Vector<Complex, 2>({ Complex(0, 0), Complex(1, 0) }), Matrix<Complex, 2, 2>::Identity());
    }
}

class Mixed_Qubit : public Qubit {
public:
    Mixed_Qubit(const Qubit& q) : rho{ q.density_matrix() } {}

    template<std::size_t N>

    explicit Mixed_Qubit(const std::array<Qubit, N>& q, const std::array<double, N>& p) {
        for (std::size_t i = 0; i < N; i++) {
            rho += p[i] * (q[i].get_state() * q[i].get_state().adjoint());
        }
    }

    inline Mixed_Qubit(const Matrix<Complex, 2, 2>& rhoo) : rho{rhoo}{}


    inline void operator*(const Matrix<Complex, 2, 2>& U) { rho = U * rho * U.adjoint(); }

    inline Matrix<Complex, 2, 2> density_matrix() { return rho; }

   
private:
    Matrix<Complex, 2, 2> rho;
};

template<std::size_t N>
constexpr size_t pow2 = (1 << N);

namespace Q_Gates {
    static const Matrix2cd X = (Matrix2cd() << Complex(0, 0), Complex(1, 0), Complex(1, 0), Complex(0, 0)).finished();
    static const Matrix2cd Z = (Matrix2cd() << Complex(1, 0), Complex(0, 0), Complex(0, 0), Complex(-1, 0)).finished();
    static const Matrix2cd Y = (Matrix2cd() << Complex(0, 0), Complex(0, -1), Complex(0, 1), Complex(0, 0)).finished();
    static const Matrix2cd H = (Matrix2cd() << Complex(1 / sqrt(2), 0), Complex(1 / sqrt(2), 0), Complex(1 / sqrt(2), 0), Complex(-1 / sqrt(2), 0)).finished();
    static const Matrix2cd I = (Matrix2cd() << Complex(1, 0), Complex(0, 0), Complex(0, 0), Complex(0, 1)).finished();

    inline Matrix2cd Rot_X(const double& theta) { return cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * X; }
    inline Matrix2cd Rot_Y(const double& theta) { return cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * Y; }
    inline Matrix2cd Rot_Z(const double& theta) { return cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * Z; }

    inline Matrix2cd R_k(const size_t& k) {
        return (Matrix2cd() << Complex(1, 0), Complex(0, 0), Complex(0, 0), std::exp(Complex(0, (2 * PI) / (1 << k)))).finished();
        
    }
    inline Matrix2cd Rot_N(const Vector3d& n, const double& theta) {
        return cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * (n[0] * X + n[1] * Y + n[2] * Z);
    }

    inline Matrix2cd Gate(const Vector3d& n, const double& theta, const double& alpha) {
        return std::exp(Complex(0, alpha)) * (cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * (n[0] * X + n[1] * Y + n[2] * Z));
    }

    inline void cnot(Qubit& control, Qubit& target) {
        bool pure = (true);
        if (control[1] != Complex(0, 0)) {
            target = Qubit({ target[1], target[0] }, target.get_Bases());
        }
    }

    static const Matrix4cd CNOT = (Matrix4cd() <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 0).finished();

}

template <std::size_t N>
class Qubits {
    static constexpr size_t D = pow2<N>;

public:
    inline Qubits() : state(Vector<Complex, D>::Zero()), Bases(Matrix<Complex, D, D>::Identity()) {
        state(0) = 1;
    }

    Qubits(const Vector<Complex, D>& v, const Matrix<Complex, D, D>& b) : state(v), Bases(b) {
        normalize();
    }

    inline void operator>>(const Matrix<Complex, D, D>& P) {
        Bases = P;
        state = P.inverse() * state;
    }

    inline void normalize() {
        Complex norm = state.norm();
        if (norm != Complex(0, 0)) {
            state = state / norm;
        }
        else {
            state = Vector<Complex, D>::Zero();
        }
    }

    inline Mixed_Qubit operator[](const size_t& i) {
        Matrix<Complex, 2, 2> rho_i = Matrix<Complex, 2, 2>::Zero();
        int num_qubits = N; // total number of qubits
        int block_size = 1 << i; // 2^i, size of the block for the i-th qubit

        for (int j = 0; j < (1 << (num_qubits - 1)); ++j) {
            // b0 is the index where the i-th qubit is 0
            int b0 = (j & ((1 << i) - 1)) + ((j >> i) << (i + 1));
            // b1 is the index where the i-th qubit is 1
            int b1 = b0 + block_size;

            rho_i(0, 0) += std::conj(state(b0)) * state(b0);
            rho_i(0, 1) += std::conj(state(b0)) * state(b1);
            rho_i(1, 0) += std::conj(state(b1)) * state(b0);
            rho_i(1, 1) += std::conj(state(b1)) * state(b1);
        }

        return Mixed_Qubit(rho_i);
    }

    inline Qubits operator+(const Qubits& other) {
        return Qubits(state + other.state, Bases).normalize();
    }

    inline Qubits operator-(const Qubits& other) {
        return Qubits(state - other.state, Bases).normalize();
    }

    //unsupported currently
    inline Qubits<N + 1> operator^(const Qubit& q) {
        return Qubits<N + 1>(kroneckerProduct(state, q.get_state()).eval(), kroneckerProduct(Bases, q.get_Bases()).eval());
    }

    inline Complex operator*(const Qubits& other) {
        if (Bases != other.Bases) {
            throw std::invalid_argument("Cannot compute inner product of qubits with different bases.");
        }
        return state.dot(other.state);
    }

    void applyGate(const Eigen::Matrix<std::complex<double>, 2, 2>& gate, const size_t& qubit_index) {
        const size_t step = 1 << qubit_index;

        for (size_t i = 0; i < D; i += 2 * step) {
            for (size_t j = 0; j < step; ++j) {
                size_t idx1 = i + j;
                size_t idx2 = idx1 + step;

                // Cache the current values
                Eigen::Vector2cd state_segment;
                state_segment(0) = state(idx1);
                state_segment(1) = state(idx2);

                // Apply the gate
                Eigen::Vector2cd result = gate * state_segment;

                // Update the state vector
                state(idx1) = result(0);
                state(idx2) = result(1);
            }
        }
    }

    void applyGate(const MatrixXcd& gate, const size_t& start_index, const size_t& end_index) {
        const size_t k = end_index - start_index + 1; // Number of qubits the gate acts on
        const size_t gate_dim = 1 << k; // 2^k
        const size_t step = 1 << start_index;
        const size_t end_step = 1 << (end_index + 1);

        for (size_t i = 0; i < D; i += end_step) {
            for (size_t j = 0; j < step; ++j) {
                for (size_t l = 0; l < gate_dim; ++l) {
                    size_t idx1 = i + j + (l & ((1 << k) - 1)) * step;

                    Eigen::Map<VectorXcd> state_segment(&state(idx1), gate_dim);

                    // Apply the gate
                    state_segment = gate * state_segment;
                }
            }
        }
    }
    void applyCGate(const Matrix<Complex, 2, 2>& U, const size_t& c_index, const size_t& t_index) {
        if (c_index == t_index) return; // Control and target can't be the same

        size_t control_mask = 1ULL << c_index;
        size_t target_mask = 1ULL << t_index;

        for (size_t i = 0; i < D; ++i) {
            // Only process states where the control qubit is 1
            if (i & control_mask) {
                size_t j = i ^ target_mask; // Flip the target qubit
                size_t target_bit = (i >> t_index) & 1; // Extract the target qubit's value (0 or 1)

                // Apply the controlled-U gate
                Complex new_i = U(target_bit, 0) * state[i] + U(target_bit, 1) * state[j];
                Complex new_j = U(target_bit ^ 1, 0) * state[i] + U(target_bit ^ 1, 1) * state[j];

                // Update the state vector
                state[i] = new_i;
                state[j] = new_j;
            }
        }
    }

    void applyCGate(const MatrixXcd& gate, const size_t& c_index, const size_t& start_index, const size_t& end_index) {
        if (c_index >= start_index && c_index <= end_index) {
            throw std::invalid_argument("Control qubit cannot overlap with the target qubits.");
        }

        const size_t k = end_index - start_index + 1; // Number of target qubits
        const size_t gate_dim = 1 << k; // Dimension of the gate (2^k)
        const size_t control_mask = 1ULL << c_index;
        const size_t step = 1 << start_index;
        const size_t end_step = 1 << (end_index + 1);

        for (size_t i = 0; i < D; i += end_step) {
            for (size_t j = 0; j < step; ++j) {
                // Check if the control qubit is 1
                if (i & control_mask) {
                    // Extract the relevant subspace of the state vector
                    Eigen::VectorXcd subspace(gate_dim);
                    for (size_t l = 0; l < gate_dim; ++l) {
                        size_t idx = i + j + (l * step);
                        subspace(l) = state(idx);
                    }

                    // Apply the gate
                    Eigen::VectorXcd new_subspace = gate * subspace;

                    // Update the state vector
                    for (size_t l = 0; l < gate_dim; ++l) {
                        size_t idx = i + j + (l * step);
                        state(idx) = new_subspace(l);
                    }
                }
            }
        }
    }

    void applyQFT() {
        // Apply the Quantum Fourier Transform
        for (size_t i = 0; i < N; ++i) {
            // Apply the Hadamard gate to qubit i
            applyGate(Q_Gates::H, i);

            // Apply controlled R_k gates
            for (size_t j = i + 1; j < N; ++j) {
                size_t k = j - i + 1;
                applyCGate(Q_Gates::R_k(k), j, i);
            }
        }

        // Swap qubits to reverse the order
        for (size_t i = 0; i < N / 2; ++i) {
            applyGate(Q_Gates::SWAP, i, N - i - 1);
        }
    }

    void applyIQFT() {
        for (size_t i = 0; i < N / 2; ++i) {
            applyGate(Q_Gates::SWAP, i, N - i - 1);
        }

        // Apply the Quantum Fourier Transform
        for (size_t i = 0; i < N; ++i) {
            // Apply controlled R_k gates
            for (size_t j = i + 1; j < N; ++j) {
                size_t k = j - i + 1;
                applyCGate(Q_Gates::R_k(k).adjoint(), j, i);
            }
            // Apply the Hadamard gate to qubit i
            applyGate(Q_Gates::H, i);
        }
    }

    inline void operator* (const Matrix4cd& U) { state = U * state;}
    
    inline Vector<Complex, D> state_vector() const { return state; }
    inline Matrix<Complex, D, D> density_matrix() const {s
        return state * state.adjoint();
    }

    void measureQubit(const size_t& qubit_index) {
        Qubit q = (*this)[qubit_index];
        q.measure_qubit();
        bool measured_0 = (q[0] == Complex(1,0));
        collapseSystem(qubit_index, measured_0);
    }

private:
    Vector<Complex, D> state;
    Matrix<Complex, D, D> Bases;

    void collapseSystem(const size_t& qubit_index, const bool &measured_0) {

        Vector<Complex, D> new_state = Vector<Complex, D>::Zero();

        // Iterate through the state vector
        for (size_t i = 0; i < D; ++i) {
            // Extract the value of the qubit_index-th bit in the state vector index
            bool qubit_value = (i >> qubit_index) & 1;

            // Keep only the components corresponding to the measurement outcome
            if ((measured_0 && qubit_value == 0) || (!measured_0 && qubit_value == 1)) {
                new_state[i] = state[i];
            }
        }

        // Normalize the new state vector to ensure it has unit norm
        Complex norm = new_state.norm();
        if (norm != Complex(0, 0)) {
            state = new_state / norm;
        }
        else {
            throw std::runtime_error("Collapse resulted in zero norm, something went wrong.");
        }
    }

};

void test_1() {

    // Define the CNOT gate as a 4x4 matrix
    Eigen::Matrix<Complex, 4, 4> CNOT = Eigen::Matrix<Complex, 4, 4>::Zero();
    CNOT(0, 0) = 1;
    CNOT(1, 1) = 1;
    CNOT(2, 3) = 1;
    CNOT(3, 2) = 1;

    // Initial state |01> (2-qubit system, so state vector has size 4)
    Eigen::Vector4cd initial_state;
    initial_state << 0, 0, 0, 1; // |11> state

    // Identity matrix for basis (4x4 for 2 qubits)
    Eigen::Matrix<Complex, 4, 4> identity = Eigen::Matrix<Complex, 4, 4>::Identity();

    // Create Qubits object with initial state
    Qubits<2> qubits(initial_state, identity);

    // Define a 2x2 identity gate (no-op)
    Matrix2cd identity_gate = Matrix2cd::Identity();

    // Apply the CNOT gate using applyCGate
    qubits.applyCGate(identity_gate, 1, 0 );

    // Output the resulting state
    std::cout << "State after applying controlled gate:\n" << qubits.state_vector() << std::endl; //should be |10>

}



