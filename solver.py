import math

import numpy as np
from numpy import linalg
import norm
from tabulate import tabulate
import gauss


def describe(A, b, x):
    print_A = tabulate(A, tablefmt="fancy_grid")
    print_b = tabulate(b.T, tablefmt="fancy_grid")
    print_x = tabulate(x.T, tablefmt="fancy_grid")

    print('Матрица A')
    print(print_A)
    print('Вектор b')
    print(print_b)
    print('Решение системы')
    print(print_x)
    print("Число обсуловленности: ", norm.cond(A))
    print("Точность решения: ", norm.vnorm_3(np.matmul(A, x) - b))


def method_sweep(A, b, verbose=False):
    """
    :param verbose: Вывести подробную информацию
    :param A: Трехдиагональной матрица коэффициентов размером n на n
    :param b: Вектор-столбец размером n на 1
    :return: X - вектор-столбец решение системы размером n на 1
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    n = A.shape[0]

    # Получаем диагональные элементы
    diag_upper = np.diagonal(A, offset=1)
    diag_main = -np.diagonal(A)
    diag_lower = np.diagonal(A, offset=-1)

    P = np.zeros(n)
    Q = np.zeros(n)

    P[0] = diag_upper[0] / diag_main[0]
    Q[0] = -b[0] / diag_main[0]

    for i in range(1, n - 1):
        div = diag_main[i] - diag_lower[i - 1] * P[i - 1]
        P[i] = diag_upper[i] / div
        Q[i] = (diag_lower[i - 1] * Q[i - 1] - b[i]) / div

    x = np.zeros((n, 1))

    x[-1] = (diag_upper[-1] * Q[-2] - b[-1] - b[-1]) / (diag_main[-1] - diag_lower[-1] * P[-2])
    for i in reversed(range(n - 1)):
        x[i] = P[i] * x[i + 1] + Q[i]

    if verbose:
        describe(A, b, x)

    return x


def method_LU(A, b, verbose=False):
    """
     :param verbose: Вывести подробную информацию
     :param A: Квадратная матрица коэффициентов размером n на n
     :param b: Вектор-столбец размером n на 1
     :return: X - вектор-столбец решение системы размером n на 1
     """

    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    n = A.shape[0]

    # Факторизация матрицы A = LU
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

    # Ly = b
    y = gauss.backward_lower(L, b)

    # Ux = y
    x = gauss.backward_upper(U, y)

    if verbose:
        describe(A, b, x)

    return x


def method_UU(A, b, verbose=False):
    """
         :param verbose: Вывести подробную информацию
         :param A: Симметричная положительно опредленная квадратная матрица коэффициентов размером n на n
         :param b: Вектор-столбец размером n на 1
         :return: X - вектор-столбец решение системы размером n на 1
         """
    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if not np.array_equal(A, A.T):
        raise ValueError('Матрица A несимметричная')

    if not np.all(np.linalg.eigvals(A) > 0):
        raise ValueError('Матрица A является неопределенной')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    n = A.shape[0]

    # Факторизация матрицы A = U^T * U
    # U -  верхняя треугольная матрица
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

    # Ly = b
    y = gauss.backward_lower(U.T, b)

    # Ux = y
    x = gauss.backward_upper(U, y)

    if verbose:
        describe(A, b, x)

    return x






