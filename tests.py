import unittest
import gauss
import iteralg
import norm
import numpy as np
import solver


class TestingVectorAndMatrixNorm(unittest.TestCase):

    # Пример вычислений векторной нормы из лекции (слайд 18)
    def test_vnorm(self):
        x = np.array([1, -2, 3, -4]).T
        norm_1 = norm.vnorm_1(x)
        self.assertEqual(norm_1, 4)

        norm_2 = norm.vnorm_2(x)
        self.assertEqual(norm_2, 10)

        norm_3 = norm.vnorm_3(x)
        self.assertTrue((norm_3 - np.sqrt(30)) < 10e-6)

    # Пример вычислений матричной нормы из лекции (слайд 24)
    def test_mnorm(self):
        x = np.array([1, -2, 3, -4]).T
        norm_1 = norm.vnorm_1(x)
        self.assertEqual(norm_1, 4)

        norm_2 = norm.vnorm_2(x)
        self.assertEqual(norm_2, 10)

        norm_3 = norm.vnorm_3(x)
        self.assertTrue((norm_3 - np.sqrt(30)) < 10e-6)


class TestingMethodGauss(unittest.TestCase):

    # Пример решения СЛАУ из лекции (слайд 49)
    def test_gauss_method_1(self):
        A = np.array([[2, 1, 4], [3, 2, 1], [1, 3, 3]], dtype=float)
        b = np.array([[16, 10, 16]], dtype=float)
        x = gauss.method_gauss_1(A, b.T)

        new_b = np.matmul(A, x)
        self.assertTrue(np.array_equal(new_b, b.T))

    # Пример решения СЛАУ из лекции (слайд 52)
    def test_gauss_method_2(self):
        A = np.array([[-3, 2.099, 6], [10, -7, 0], [5, -1, 5]], dtype=float)
        b = np.array([[3.901, 7, 6]], dtype=float)
        x = gauss.method_gauss_2(A, b.T)

        new_b = np.matmul(A, x)
        self.assertTrue(norm.vnorm_3(new_b - b.T) < 1e-6)


class TestingOtherSolvers(unittest.TestCase):

    # Пример решения СЛАУ из лекции (слайд 65)
    def test_method_sweep(self):
        A = np.array([[5, 3, 0, 0], [3, 6, 1, 0], [0, 1, 4, -2], [0, 0, 1, -3]], dtype=float)
        b = np.array([[8, 10, 3, -2]], dtype=float).T
        x = solver.method_sweep(A, b, verbose=False)

        new_b = np.matmul(A, x)
        self.assertTrue(norm.vnorm_3(new_b - b) < 1)

    # Пример решения СЛАУ из лекции (слайд 84)
    def test_method_LU(self):
        A = np.array([[2, 1, 4], [3, 2, 1], [1, 3, 3]], dtype=float)
        b = np.array([[1, 3, 3]], dtype=float).T
        x = solver.method_LU(A, b, verbose=False)

        new_b = np.matmul(A, x)
        self.assertTrue(norm.vnorm_3(new_b - b) < 1e-3)

    # Пример решения СЛАУ из лекции (слайд 94)
    def test_method_UU(self):
        A = np.array([[2, 1, 4], [1, 1, 3], [4, 3, 14]], dtype=float)
        b = np.array([[16, 12, 52]], dtype=float).T
        x = solver.method_UU(A, b, verbose=False)

        new_b = np.matmul(A, x)
        self.assertTrue(norm.vnorm_3(new_b - b) < 1e-3)

    # Пример решения СЛАУ из лекции (слайд 112)
    def test_method_simple_iterations(self):
        A = np.array([[10, 1, 1], [2, 10, 1], [2, 2, 10]], dtype=float)
        b = np.array([[2, 2, 10]], dtype=float).T
        x = iteralg.method_simple_iterations(A, b)

        new_b = np.matmul(A, x)
        self.assertTrue(norm.vnorm_3(new_b - b) < 1)

        # Пример решения СЛАУ из лекции (слайд 112)
    def test_method_seidel(self):
        A = np.array([[10, 1, 1], [2, 10, 1], [2, 2, 10]], dtype=float)
        b = np.array([[2, 2, 10]], dtype=float).T
        x = iteralg.method_seidel(A, b)

        new_b = np.matmul(A, x)
        self.assertTrue(norm.vnorm_3(new_b - b) < 1)

    def test_inv_matrix(self):
        A = np.array([[1, 2, 1], [0, 1, 0], [0, 2, 2]], dtype=float)
        U = iteralg.inv(A, verbose=True)

        # self.assertTrue(norm.mnorm_2(E) < 1)

    def test_spectral_radius(self):
        A = np.array([[5, 1, 2], [1, 4, 1], [2, 1, 3]], dtype=float)
        l, x = iteralg.spectral_radius(A, eps=0.1, verbose=True)

        # print(l, x, np.linalg.eig(A))

if __name__ == '__main__':
    unittest.main(verbosity=3)
