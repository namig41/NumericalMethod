import math
import numpy as np
from numpy import linalg
from tabulate import tabulate


def get_LU(A, verbose=False):
    """
    Разложение матрицы A = LU, где L - нижняя треугольная матрица, U -  верхняя треугольная матрица
    :param A: Квадртатная матрица n x n
    :return: L - нижняя треугольная матрица, U -  верхняя треугольная матрица,
    """

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    n = A.shape[0]

    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            U[0, i] = A[0, i]
            L[i, 0] = A[i, 0] / U[0, 0]

            sum = 0
            for k in range(i):
                sum += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - sum
            if i > j:
                L[j, i] = 0
            else:
                sum = 0
                for k in range(i):
                    sum += L[j, k] * U[k, i]
                L[j, i] = (A[j, i] - sum) / U[i, i]

    if verbose:
        print_L = tabulate(L, tablefmt="fancy_grid")
        print_U = tabulate(U, tablefmt="fancy_grid")

        print('Матрица L')
        print(print_L)
        print('Вектор U')
        print(print_U)

    return L, U


def get_UU(A, verbose=False):

    """
    Разложение матрицы A = U^T*U, где  U -  верхняя треугольная матрица
    :param A: Квадртатная матрица n x n
    :return: U -  верхняя треугольная матрица,
    """

    if not np.array_equal(A, A.T):
        raise ValueError('Матрица A несимметричная')

    if not np.all(np.linalg.eigvals(A) > 0):
        raise ValueError('Матрица A является неопределенной')

    n = A.shape[0]

    U = np.zeros((n, n))
    U[0, 0] = math.sqrt(A[0, 0])
    for j in range(1, n):
        U[0, j] = A[j, 0] / U[0, 0]
    for i in range(1, n):
        for j in range(i, n):
            if i == j:
                sum = 0
                for k in range(j):
                    sum += U[k, j] ** 2
                U[i, i] = math.sqrt(A[i, i] - sum)
            else:
                sum = 0
                for k in range(j):
                    sum += U[k, i] * U[k, j]
                U[i, j] = 1 / U[i, i] * (A[i, j] - sum)

    if verbose:
        print_U = tabulate(U, tablefmt="fancy_grid")
        print_UT = tabulate(U.T, tablefmt="fancy_grid")

        print('Матрица U')
        print(print_U)
        print('Вектор UT')
        print(print_UT)

    return U
