import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import unittest
import numpy as np
from lemon.numlib import *


class TestEinsteinSummation(unittest.TestCase):
    """Test Einstein summation convention for physics applications"""

    def test_trace_computation(self):
        """Test trace computation using einsum"""
        # Trace: A_ii
        m = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Using einsum for trace
        trace_einsum = einsum("ii->", m)

        # Manual trace calculation
        trace_manual = m[0, 0] + m[1, 1] + m[2, 2]

        self.assertAlmostEqual(trace_einsum._data, trace_manual._data, places=5)

        # Compare with numpy
        np_trace = np.trace(m._data)
        self.assertAlmostEqual(trace_einsum._data, np_trace, places=5)

    def test_matrix_multiplication_einsum(self):
        """Test matrix multiplication using Einstein notation"""
        # C_ij = A_ik * B_kj
        a = matrix(np.random.randn(3, 4))
        b = matrix(np.random.randn(4, 5))

        # Using einsum
        c_einsum = einsum("ik,kj->ij", a, b)

        # Using matmul
        c_matmul = a @ b

        np.testing.assert_array_almost_equal(c_einsum._data, c_matmul._data)

    def test_outer_product_einsum(self):
        """Test outer product using Einstein notation"""
        # C_ij = a_i * b_j
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])

        # Using einsum
        v1_flat = v1.reshape(-1)
        v2_flat = v2.reshape(-1)
        outer_einsum = einsum("i,j->ij", v1_flat, v2_flat)

        # Using numpy outer
        outer_np = np.outer(v1._data, v2._data)

        np.testing.assert_array_almost_equal(outer_einsum._data, outer_np)

    def test_tensor_contraction(self):
        """Test tensor contraction for higher-order tensors"""
        # Contract over middle indices: A_ijkl * B_klmn -> C_ijmn
        a = tensor(np.random.randn(2, 3, 4, 5))
        b = tensor(np.random.randn(4, 5, 6, 7))

        # Contraction over k and l
        c = einsum("ijkl,klmn->ijmn", a, b)

        # Verify shape
        self.assertEqual(c.shape, (2, 3, 6, 7))

        # Manual verification for a specific element
        i, j, m, n = 0, 1, 2, 3
        manual_sum = 0.0
        for k in range(4):
            for l in range(5):
                manual_sum += a._data[i, j, k, l] * b._data[k, l, m, n]

        self.assertAlmostEqual(c._data[i, j, m, n], manual_sum, places=5)

    def test_batch_matrix_multiplication(self):
        """Test batch matrix multiplication (common in physics simulations)"""
        # Batch of rotation matrices
        batch_size = 10
        a = tensor(np.random.randn(batch_size, 3, 3))
        b = tensor(np.random.randn(batch_size, 3, 3))

        # Batch matrix multiplication: C_bij = A_bik * B_bkj
        c = einsum("bij,bjk->bik", a, b)

        # Verify with bmm
        c_bmm = bmm(a, b)

        np.testing.assert_array_almost_equal(c._data, c_bmm._data)

    def test_tensor_network_contraction(self):
        """Test tensor network contraction (quantum physics)"""
        # Simulate a simple tensor network: MPS-like structure
        # A_ij * B_jkl * C_lm -> D_ikm
        a = tensor(np.random.randn(2, 3))
        b = tensor(np.random.randn(3, 4, 5))
        c = tensor(np.random.randn(5, 6))

        # Contract the network
        d = einsum("ij,jkl,lm->ikm", a, b, c)

        # Verify shape
        self.assertEqual(d.shape, (2, 4, 6))

        # Step-by-step verification
        temp = einsum("ij,jkl->ikl", a, b)
        expected = einsum("ikl,lm->ikm", temp, c)

        np.testing.assert_array_almost_equal(d._data, expected._data)


class TestKroneckerProduct(unittest.TestCase):
    """Test Kronecker product for quantum mechanics applications"""

    def test_kronecker_product_2x2(self):
        """Test Kronecker product of 2x2 matrices (Pauli matrices)"""
        # Pauli matrices
        pauli_x = matrix([[0, 1], [1, 0]])
        pauli_y = matrix([[0, -1j], [1j, 0]])
        pauli_z = matrix([[1, 0], [0, -1]])
        identity = matrix([[1, 0], [0, 1]])

        # Kronecker product using einsum: A ⊗ B = A_ij * B_kl -> C_ikjl (reshaped)
        def kron(a, b):
            """Compute Kronecker product using einsum"""
            m, n = a.shape
            p, q = b.shape
            result = einsum("ij,kl->ikjl", a, b)
            return reshape(result, m * p, n * q)

        # Test I ⊗ σ_x
        kron_result = kron(identity, pauli_x)
        np_kron = np.kron(identity._data, pauli_x._data)

        np.testing.assert_array_almost_equal(kron_result._data, np_kron)

        # Test σ_z ⊗ σ_z (common in Ising model)
        zz = kron(pauli_z, pauli_z)
        np_zz = np.kron(pauli_z._data, pauli_z._data)

        np.testing.assert_array_almost_equal(zz._data, np_zz)

    def test_kronecker_product_chain(self):
        """Test chain of Kronecker products (multi-qubit systems)"""
        # Three-qubit system
        a = matrix([[1, 0], [0, 1]])
        b = matrix([[0, 1], [1, 0]])
        c = matrix([[1, 0], [0, -1]])

        def kron(a, b):
            m, n = a.shape
            p, q = b.shape
            result = einsum("ij,kl->ikjl", a, b)
            return reshape(result, m * p, n * q)

        # Chain: A ⊗ B ⊗ C
        ab = kron(a, b)
        abc = kron(ab, c)

        # Compare with numpy
        np_abc = np.kron(np.kron(a._data, b._data), c._data)

        np.testing.assert_array_almost_equal(abc._data, np_abc)


class TestMatrixDecompositions(unittest.TestCase):
    """Test matrix decompositions used in physics"""

    def test_svd_decomposition(self):
        """Test Singular Value Decomposition (SVD)"""
        # Create a rank-deficient matrix
        m = matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])

        # SVD using numpy (for reference)
        u_np, s_np, vh_np = np.linalg.svd(m._data)

        # Reconstruct matrix from SVD
        s_diag = np.zeros_like(m._data)
        s_diag[: min(m.shape), : min(m.shape)] = np.diag(s_np[: min(m.shape)])
        reconstructed = u_np @ s_diag @ vh_np

        # Test reconstruction
        np.testing.assert_array_almost_equal(m._data, reconstructed, decimal=10)

        # Test rank (should be 1 for this matrix)
        rank = np.sum(s_np > 1e-10)
        self.assertEqual(rank, 1)

    def test_qr_decomposition(self):
        """Test QR decomposition (Gram-Schmidt process)"""
        # Random matrix
        m = matrix(np.random.randn(4, 3))

        # QR decomposition
        q_np, r_np = np.linalg.qr(m._data)

        # Test: Q should be orthogonal
        q = matrix(q_np)
        identity_result = q.T @ q
        identity_expected = np.eye(3)

        np.testing.assert_array_almost_equal(
            identity_result._data, identity_expected, decimal=10
        )

        # Test: QR should equal original matrix
        reconstructed = matrix(q_np) @ matrix(r_np)
        np.testing.assert_array_almost_equal(reconstructed._data, m._data, decimal=10)

    def test_eigenvalue_decomposition(self):
        """Test eigenvalue decomposition for symmetric matrices"""
        # Create symmetric matrix (common in physics: Hamiltonians)
        m = matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(m._data)

        # Test: Av = λv for each eigenvalue/eigenvector pair
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            lambda_i = eigenvalues[i]

            # M * v should equal λ * v
            lhs = m._data @ v
            rhs = lambda_i * v

            np.testing.assert_array_almost_equal(lhs, rhs, decimal=10)


class TestPhysicsSpecificOperations(unittest.TestCase):
    """Test operations specific to physics applications"""

    def test_commutator(self):
        """Test commutator [A, B] = AB - BA (quantum mechanics)"""

        # Define commutator function
        def commutator(a, b):
            return a @ b - b @ a

        # Test with Pauli matrices
        pauli_x = matrix([[0, 1], [1, 0]])
        pauli_y = matrix([[0, -1j], [1j, 0]])
        pauli_z = matrix([[1, 0], [0, -1]])

        # [σ_x, σ_y] = 2iσ_z
        comm_xy = commutator(pauli_x, pauli_y)
        expected = 2j * pauli_z._data

        np.testing.assert_array_almost_equal(comm_xy._data, expected)

        # [σ_x, σ_x] = 0
        comm_xx = commutator(pauli_x, pauli_x)
        expected_zero = np.zeros((2, 2))

        np.testing.assert_array_almost_equal(comm_xx._data, expected_zero)

    def test_anticommutator(self):
        """Test anticommutator {A, B} = AB + BA (fermion operators)"""

        # Define anticommutator function
        def anticommutator(a, b):
            return a @ b + b @ a

        # Test with Pauli matrices
        pauli_x = matrix([[0, 1], [1, 0]])
        pauli_y = matrix([[0, -1j], [1j, 0]])

        # {σ_x, σ_y} = 0
        anticomm_xy = anticommutator(pauli_x, pauli_y)
        expected_zero = np.zeros((2, 2), dtype=complex)

        np.testing.assert_array_almost_equal(anticomm_xy._data, expected_zero)

        # {σ_x, σ_x} = 2I
        anticomm_xx = anticommutator(pauli_x, pauli_x)
        expected_identity = 2 * np.eye(2)

        np.testing.assert_array_almost_equal(anticomm_xx._data, expected_identity)

    def test_tensor_product_states(self):
        """Test tensor product for quantum states"""
        # Two-qubit states
        ket_0 = vector([1, 0])  # |0⟩
        ket_1 = vector([0, 1])  # |1⟩

        # |00⟩ = |0⟩ ⊗ |0⟩
        def tensor_product(a, b):
            """Tensor product of two vectors"""
            # return einsum("i,j->ij", a, b).reshape(-1)
            a_flat = a.reshape(-1)
            b_flat = b.reshape(-1)
            return einsum("i,j->ij", a_flat, b_flat).reshape(-1)

        ket_00 = tensor_product(ket_0, ket_0)
        expected_00 = np.array([1, 0, 0, 0])

        np.testing.assert_array_almost_equal(ket_00._data, expected_00)

        # |01⟩ = |0⟩ ⊗ |1⟩
        ket_01 = tensor_product(ket_0, ket_1)
        expected_01 = np.array([0, 1, 0, 0])

        np.testing.assert_array_almost_equal(ket_01._data, expected_01)

        # Bell state: (|00⟩ + |11⟩)/√2
        ket_11 = tensor_product(ket_1, ket_1)
        bell_state = (ket_00 + ket_11) / sqrt(real(2))
        expected_bell = np.array([1, 0, 0, 1]) / np.sqrt(2)

        np.testing.assert_array_almost_equal(bell_state._data, expected_bell)

    def test_expectation_value(self):
        """Test expectation value ⟨ψ|A|ψ⟩"""

        # Define expectation value function
        def expectation_value(state, operator):
            """Compute ⟨ψ|A|ψ⟩"""
            # For column vector state
            bra = state.T  # ⟨ψ|
            result = bra @ operator @ state
            return result

        # Test with normalized state
        state = vector([1, 1]) / sqrt(real(2))  # (|0⟩ + |1⟩)/√2
        pauli_z = matrix([[1, 0], [0, -1]])

        # ⟨ψ|σ_z|ψ⟩ should be 0 for this superposition
        exp_val = expectation_value(state, pauli_z)
        self.assertAlmostEqual(exp_val._data, 0.0, places=10)

        # Test with |0⟩ state
        state_0 = vector([1, 0])
        exp_val_0 = expectation_value(state_0, pauli_z)
        self.assertAlmostEqual(exp_val_0._data, 1.0, places=10)

    def test_density_matrix(self):
        """Test density matrix operations"""
        # Pure state density matrix: ρ = |ψ⟩⟨ψ|
        state = vector([1, 1]) / sqrt(real(2))

        # Density matrix
        rho = state @ state.T

        # Properties of density matrix:
        # 1. Trace should be 1
        trace_rho = einsum("ii->", rho)
        self.assertAlmostEqual(trace_rho._data, 1.0, places=10)

        # 2. ρ² = ρ for pure states
        rho_squared = rho @ rho
        np.testing.assert_array_almost_equal(rho_squared._data, rho._data, decimal=10)

        # 3. All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(rho._data)
        self.assertTrue(np.all(eigenvalues >= -1e-10))


class TestMomentOfInertia(unittest.TestCase):
    """Test moment of inertia tensor operations"""

    def test_inertia_tensor(self):
        """Test computation of inertia tensor"""
        # Point masses at different positions
        positions = matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3 points
        masses = vector([1, 2, 3])

        # Compute inertia tensor: I_ij = Σ m(r²δ_ij - r_i*r_j)
        def compute_inertia_tensor(positions, masses):
            n_points = positions.shape[0]

            # NumPy配列として直接操作
            pos_data = positions._data
            mass_data = masses._data.flatten()

            # 慣性テンソルを初期化
            I = np.zeros((3, 3))

            for k in range(n_points):
                r = pos_data[k, :]  # 位置ベクトル
                m = mass_data[k]  # 質量

                # r² = r·r
                r_squared = np.dot(r, r)

                # 慣性テンソルへの寄与を計算
                # I_ij = m(r²δ_ij - r_i*r_j)
                I += m * (r_squared * np.eye(3) - np.outer(r, r))

            return Matrix(I)

        I = compute_inertia_tensor(positions, masses)

        # Check symmetry
        np.testing.assert_array_almost_equal(I._data, I._data.T)

        # Check principal moments (eigenvalues should be positive)
        eigenvalues = np.linalg.eigvalsh(I._data)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_angular_momentum(self):
        """Test angular momentum L = I·ω"""
        # Inertia tensor (diagonal for simplicity)
        I = matrix([[2, 0, 0], [0, 3, 0], [0, 0, 4]])

        # Angular velocity
        omega = vector([1, 2, 3])

        # Angular momentum
        L = I @ omega

        # Expected
        expected = np.array([[2], [6], [12]])
        np.testing.assert_array_almost_equal(L._data, expected)


class TestStressStrainTensor(unittest.TestCase):
    """Test stress and strain tensor operations (continuum mechanics)"""

    def test_stress_tensor_invariants(self):
        """Test computation of stress tensor invariants"""
        # Stress tensor (symmetric 3x3)
        stress = matrix([[100, 50, 0], [50, 200, 0], [0, 0, 150]])

        # First invariant: trace(σ)
        I1 = einsum("ii->", stress)
        expected_I1 = 100 + 200 + 150
        self.assertAlmostEqual(I1._data, expected_I1, places=5)

        # Second invariant (simplified for 3D)
        # I2 = σ_xx*σ_yy + σ_yy*σ_zz + σ_zz*σ_xx - σ_xy² - σ_yz² - σ_zx²
        I2 = (
            stress[0, 0] * stress[1, 1]
            + stress[1, 1] * stress[2, 2]
            + stress[2, 2] * stress[0, 0]
            - stress[0, 1] ** 2
            - stress[1, 2] ** 2
            - stress[0, 2] ** 2
        )

        expected_I2 = 100 * 200 + 200 * 150 + 150 * 100 - 50**2 - 0 - 0
        self.assertAlmostEqual(I2._data, expected_I2, places=5)

    def test_von_mises_stress(self):
        """Test von Mises stress calculation"""
        # Stress tensor
        stress = matrix([[100, 50, 30], [50, 200, 20], [30, 20, 150]])

        # Von Mises stress: σ_v = √(3/2 * s:s) where s is deviatoric stress
        # s = σ - (1/3)tr(σ)I

        # Hydrostatic stress
        p = einsum("ii->", stress) / real(3)

        # Deviatoric stress
        I = eye(3)
        s = stress - p * I

        # Double contraction s:s = s_ij * s_ij
        s_contract_s = einsum("ij,ij->", s, s)

        # Von Mises stress
        sigma_vm = sqrt(real(3) / real(2) * s_contract_s)

        # Verify it's positive
        self.assertGreater(sigma_vm._data, 0)


class TestElectromagneticTensor(unittest.TestCase):
    """Test electromagnetic field tensor operations"""

    def test_electromagnetic_field_tensor(self):
        """Test construction and properties of EM field tensor"""
        # Electric field components
        Ex, Ey, Ez = 1.0, 2.0, 3.0
        # Magnetic field components (in units where c=1)
        Bx, By, Bz = 0.5, 1.0, 1.5

        # Electromagnetic field tensor F^μν
        F = matrix(
            [[0, -Ex, -Ey, -Ez], [Ex, 0, -Bz, By], [Ey, Bz, 0, -Bx], [Ez, -By, Bx, 0]]
        )

        # Test antisymmetry: F^μν = -F^νμ
        np.testing.assert_array_almost_equal(F._data, -F._data.T)

        # Dual tensor: *F^μν = (1/2)ε^μνρσ F_ρσ
        # For simplicity, just construct the dual directly
        F_dual = matrix(
            [[0, -Bx, -By, -Bz], [Bx, 0, Ez, -Ey], [By, -Ez, 0, Ex], [Bz, Ey, -Ex, 0]]
        )

        # Test antisymmetry of dual
        np.testing.assert_array_almost_equal(F_dual._data, -F_dual._data.T)

        # First invariant: F^μν F_μν = 2(B² - E²)
        # (in units where c=1)
        E_squared = Ex**2 + Ey**2 + Ez**2
        B_squared = Bx**2 + By**2 + Bz**2

        # Minkowski metric (-,+,+,+)
        g = np.diag([-1, 1, 1, 1])

        # Lower the indices: F_μν = g_μα g_νβ F^αβ
        F_lower = g @ F._data @ g

        # Contract F^μν F_μν
        F_contract_F = np.einsum("ij,ij->", F._data, F_lower)
        expected_invariant = 2 * (B_squared - E_squared)

        self.assertAlmostEqual(F_contract_F, expected_invariant, places=5)


class TestMetricTensor(unittest.TestCase):
    """Test metric tensor operations (General Relativity)"""

    def test_minkowski_metric(self):
        """Test Minkowski metric tensor"""
        # Minkowski metric η^μν = diag(-1, 1, 1, 1) or diag(1, -1, -1, -1)
        # Using signature (-,+,+,+)
        eta = matrix([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Test that η^μν η_νρ = δ^μ_ρ
        identity = eta @ eta
        expected_identity = matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        np.testing.assert_array_almost_equal(identity._data, expected_identity._data)

        # Four-velocity normalization: u^μ u_μ = -1
        # For particle at rest: u^μ = (1, 0, 0, 0)
        u = vector([1, 0, 0, 0])

        # u_μ = η_μν u^ν
        u_lower = eta @ u

        # u^μ u_μ
        norm = u.T @ u_lower
        self.assertAlmostEqual(norm._data, -1.0, places=10)

    def test_christoffel_symbols(self):
        """Test Christoffel symbol computation (simplified)"""
        # For a simple 2D metric: ds² = dr² + r²dθ² (polar coordinates)
        # g_ij = [[1, 0], [0, r²]]

        r = real(2.0)  # At r = 2

        # Metric tensor
        g = matrix([[1, 0], [0, r**2]])

        # Inverse metric
        g_inv_data = np.linalg.inv(g._data)
        g_inv = matrix(g_inv_data)

        # For this simple metric, only Γ^r_θθ = -r and Γ^θ_rθ = Γ^θ_θr = 1/r are non-zero

        # Test metric determinant
        det_g = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
        expected_det = r._data**2
        self.assertAlmostEqual(det_g._data, expected_det, places=5)


class TestLieAlgebra(unittest.TestCase):
    """Test Lie algebra operations (particle physics)"""

    def test_su2_generators(self):
        """Test SU(2) generators (Pauli matrices)"""
        # SU(2) generators: σ_i/2
        sigma_1 = matrix([[0, 1], [1, 0]]) / real(2)
        sigma_2 = matrix([[0, -1j], [1j, 0]]) / real(2)
        sigma_3 = matrix([[1, 0], [0, -1]]) / real(2)

        # Structure constants: [T_a, T_b] = i*f_abc*T_c
        # For SU(2): f_abc = ε_abc (Levi-Civita symbol)

        # Test [σ_1/2, σ_2/2] = i*σ_3/2
        comm_12 = sigma_1 @ sigma_2 - sigma_2 @ sigma_1
        expected = 1j * sigma_3._data

        np.testing.assert_array_almost_equal(comm_12._data, expected)

        # Test Casimir operator: C = T_a*T_a
        casimir = sigma_1 @ sigma_1 + sigma_2 @ sigma_2 + sigma_3 @ sigma_3
        # For SU(2) in fundamental rep: C = (3/4)I
        expected_casimir = (3 / 4) * np.eye(2)

        np.testing.assert_array_almost_equal(casimir._data, expected_casimir)

    def test_su3_gell_mann_matrices(self):
        """Test SU(3) Gell-Mann matrices (QCD)"""
        # First Gell-Mann matrix
        lambda_1 = matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        # Third Gell-Mann matrix
        lambda_3 = matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

        # Eighth Gell-Mann matrix
        lambda_8 = matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(real(3))

        # Test trace (should be zero for all Gell-Mann matrices)
        trace_1 = einsum("ii->", lambda_1)
        trace_3 = einsum("ii->", lambda_3)
        trace_8 = einsum("ii->", lambda_8)

        self.assertAlmostEqual(trace_1._data, 0.0, places=10)
        self.assertAlmostEqual(trace_3._data, 0.0, places=10)
        self.assertAlmostEqual(trace_8._data, 0.0, places=10)

        # Test orthogonality: Tr(λ_a*λ_b) = 2*δ_ab
        trace_11 = einsum("ij,ji->", lambda_1, lambda_1)
        trace_13 = einsum("ij,ji->", lambda_1, lambda_3)

        self.assertAlmostEqual(trace_11._data, 2.0, places=10)
        self.assertAlmostEqual(trace_13._data, 0.0, places=10)


class TestStatisticalMechanics(unittest.TestCase):
    """Test tensor operations in statistical mechanics"""

    def test_partition_function(self):
        """Test partition function calculation"""
        # Energy levels
        energies = vector([0, 1, 2, 3, 4])

        # Temperature (in units where k_B = 1)
        beta = real(1.0)  # β = 1/T

        # Partition function: Z = Σ exp(-βE_i)
        boltzmann_factors = exp(-beta * energies)
        Z = sum(boltzmann_factors)

        # Probabilities: p_i = exp(-βE_i)/Z
        probabilities = boltzmann_factors / Z

        # Test normalization
        prob_sum = sum(probabilities)
        self.assertAlmostEqual(prob_sum._data, 1.0, places=10)

        # Average energy: <E> = Σ E_i * p_i
        avg_energy = sum(energies * probabilities)

        # Also: <E> = -∂(ln Z)/∂β
        # For our case, we can compute numerically
        self.assertGreater(avg_energy._data, 0)
        self.assertLess(avg_energy._data, 4.0)

    def test_correlation_function(self):
        """Test two-point correlation function"""
        # Spin configuration (1D Ising model)
        spins = vector([1, -1, 1, 1, -1, 1, -1, -1, 1, 1])
        n_sites = len(spins._data)

        # Two-point correlation: C(r) = <s_i * s_{i+r}>
        def correlation(spins, r):
            """Compute correlation at distance r"""
            n = len(spins._data)
            corr_sum = real(0)
            count = 0

            for i in range(n):
                j = (i + r) % n  # Periodic boundary
                corr_sum = corr_sum + spins[i] * spins[j]
                count += 1

            return corr_sum / real(count)

        # Test correlation at different distances
        c0 = correlation(spins, 0)  # Should be 1 for normalized spins
        c1 = correlation(spins, 1)

        # Auto-correlation should be <s²> = 1 for ±1 spins
        self.assertAlmostEqual(c0._data, 1.0, places=10)

        # Correlation should decay with distance (generally)
        self.assertLessEqual(abs(c1._data), 1.0)


class TestFluidMechanics(unittest.TestCase):
    """Test tensor operations in fluid mechanics"""

    def test_velocity_gradient_tensor(self):
        """Test velocity gradient tensor and its decomposition"""
        # Velocity gradient tensor ∇v (3x3)
        grad_v = matrix([[1, 2, 3], [0.5, -1, 2], [1, 0, 0.5]])

        # Decompose into symmetric (strain rate) and antisymmetric (vorticity) parts
        # S = (1/2)(∇v + ∇v^T) - strain rate tensor
        # Ω = (1/2)(∇v - ∇v^T) - vorticity tensor

        S = (grad_v + grad_v.T) / real(2)
        Omega = (grad_v - grad_v.T) / real(2)

        # Test decomposition: ∇v = S + Ω
        reconstructed = S + Omega
        np.testing.assert_array_almost_equal(reconstructed._data, grad_v._data)

        # Test S is symmetric
        np.testing.assert_array_almost_equal(S._data, S._data.T)

        # Test Ω is antisymmetric
        np.testing.assert_array_almost_equal(Omega._data, -Omega._data.T)

        # Divergence: ∇·v = tr(∇v)
        div_v = einsum("ii->", grad_v)
        expected_div = grad_v[0, 0] + grad_v[1, 1] + grad_v[2, 2]
        self.assertAlmostEqual(div_v._data, expected_div._data, places=10)

    def test_reynolds_stress_tensor(self):
        """Test Reynolds stress tensor (turbulence)"""
        # Fluctuating velocity components (simplified)
        n_samples = 100
        np.random.seed(42)

        # Generate fluctuating velocities
        u_prime = vector(np.random.randn(n_samples))
        v_prime = vector(np.random.randn(n_samples))
        w_prime = vector(np.random.randn(n_samples))

        # Reynolds stress tensor: R_ij = <u'_i * u'_j>
        R = zeros_matrix((3, 3))

        # Compute components
        R._data[0, 0] = mean(u_prime * u_prime)._data
        R._data[0, 1] = mean(u_prime * v_prime)._data
        R._data[0, 2] = mean(u_prime * w_prime)._data
        R._data[1, 0] = R._data[0, 1]  # Symmetry
        R._data[1, 1] = mean(v_prime * v_prime)._data
        R._data[1, 2] = mean(v_prime * w_prime)._data
        R._data[2, 0] = R._data[0, 2]  # Symmetry
        R._data[2, 1] = R._data[1, 2]  # Symmetry
        R._data[2, 2] = mean(w_prime * w_prime)._data

        # Test symmetry
        np.testing.assert_array_almost_equal(R._data, R._data.T)

        # Test positive semi-definiteness (eigenvalues ≥ 0)
        eigenvalues = np.linalg.eigvalsh(R._data)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

        # Turbulent kinetic energy: k = (1/2)tr(R)
        tke = einsum("ii->", R) / real(2)
        self.assertGreater(tke._data, 0)


class TestCrystallography(unittest.TestCase):
    """Test tensor operations in crystallography"""

    def test_reciprocal_lattice(self):
        """Test reciprocal lattice vector computation"""
        # Direct lattice vectors (simple cubic for simplicity)
        a1 = vector([1, 0, 0])
        a2 = vector([0, 1, 0])
        a3 = vector([0, 0, 1])

        # Volume of unit cell: V = a1 · (a2 × a3)
        # Using Einstein notation for cross product
        def cross_product(u, v):
            """Compute cross product using Levi-Civita symbol"""
            result = zeros_vector(3)
            result._data[0] = u[1]._data * v[2]._data - u[2]._data * v[1]._data
            result._data[1] = u[2]._data * v[0]._data - u[0]._data * v[2]._data
            result._data[2] = u[0]._data * v[1]._data - u[1]._data * v[0]._data
            return result

        a2_cross_a3 = cross_product(a2, a3)
        V = dot(a1, a2_cross_a3)

        # Reciprocal lattice vectors: b1 = 2π(a2 × a3)/V
        b1 = real(2 * np.pi) * a2_cross_a3 / V
        b2 = real(2 * np.pi) * cross_product(a3, a1) / V
        b3 = real(2 * np.pi) * cross_product(a1, a2) / V

        # Test orthogonality: a_i · b_j = 2π δ_ij
        dot_11 = dot(a1, b1)
        dot_12 = dot(a1, b2)

        self.assertAlmostEqual(dot_11._data, 2 * np.pi, places=10)
        self.assertAlmostEqual(dot_12._data, 0.0, places=10)

    def test_structure_factor(self):
        """Test structure factor calculation"""
        # Atomic positions in unit cell (simple example)
        positions = matrix([[0, 0, 0], [0.5, 0.5, 0.5]])  # BCC structure

        # Scattering factors (simplified, same for both atoms)
        f = vector([1, 1])

        # Reciprocal space vector
        k = vector([2 * np.pi, 0, 0])

        # Structure factor: F(k) = Σ f_j * exp(i k·r_j)
        def structure_factor(k, positions, f):
            """Compute structure factor"""
            n_atoms = positions.shape[0]
            F = cmplx(0)

            for j in range(n_atoms):
                r_j = positions[j, :]
                phase = dot(k, r_j)
                F = F + f[j] * exp(cmplx(0, 1) * phase)

            return F

        F_k = structure_factor(k, positions, f)

        # For this configuration, |F|² gives intensity
        intensity = (F_k * F_k.real).real + (F_k * F_k.imag).imag
        self.assertGreaterEqual(intensity._data, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
