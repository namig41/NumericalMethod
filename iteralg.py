import numpy as np
from numpy import linalg
from solver import describe
from tabulate import tabulate
import random


def method_simple_iterations(A, b, verbose=False, eps=1e-8, max_iterations=10000):
    """
    Метод простых итераций
    :param A: Квадратная матрица коэффициентов размером n на n
    :param b: Вектор-столбец размером n на 1
    :param verbose: Подробный вывод
    :param eps: Погрешность измерения
    :param max_iterations: Максимальное количество итераций
    :return: X - вектор-столбец решение системы размером n на 1
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    alpha = (np.diag(np.diagonal(A)) - A) / np.diagonal(A)
    beta = b / np.diagonal(A).reshape(b.shape)

    if linalg.norm(alpha) > 1:
        raise ValueError('Решение не сходится')

    x = np.random.random(size=b.shape)
    for k in range(max_iterations):
        x_old = x.copy()
        x = alpha @ x + beta
        if np.linalg.norm(x - x_old, ord=1) < eps:
            break

    if verbose:
        describe(A, b, x)
    return x


def method_seidel(A, b, verbose=False, eps=1e-8, max_iterations=10000):
    """
      :param A: Квадратная матрица коэффициентов размером n на n
      :param b: Вектор-столбец размером n на 1
      :param verbose: Подробный вывод
      :param eps: Погрешность измерения
      :param max_iterations: Максимальное количество итераций
      :return: X - вектор-столбец решение системы размером n на 1
      """

    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    # A = A.T @ A
    # b = A.T @ b

    n = A.shape[0]

    alpha = (np.diag(np.diagonal(A)) - A) / np.diagonal(A)
    beta = b / np.diagonal(A).reshape(b.shape)

    if linalg.norm(alpha) > 1:
        raise ValueError('Решение не сходится')

    # alpha = L + U
    L = np.tril(alpha, -1)
    U = np.triu(alpha, 0)

    inv_L = np.linalg.inv(np.eye(n) - L)

    # B1 = (E - L) ^(-1) U
    B2 = inv_L @ beta

    # B1 = (E - L) * beta
    B1 = inv_L @ U

    x = np.random.random(size=b.shape)
    for k in range(max_iterations):
        x_old = x.copy()
        x = B1 @ x + B2
        if np.linalg.norm(x - x_old) < eps:
            break

    if verbose:
        describe(A, b, x)

    return x


def method_seidel_normal(A, b, verbose=False, eps=1e-8, max_iterations=10000):
    return method_seidel(A.T @ A, A.T @ b, verbose, eps, max_iterations)


def inv(A, verbose=False, eps=1e-8, max_iterations=10000):
    """
    Итерационный метод Шульца
    :param A: Квадратная матрица коэффициентов размером n на n
    :param b: Вектор-столбец размером n на 1
    :param verbose: Подробный вывод
    :param eps: Погрешность измерения
    :param max_iterations: Максимальное количество итераций
    :return: X - вектор-столбец решение системы размером n на 1
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    n = A.shape[0]
    wa, wb = linalg.eigh(A @ A.T)
    b = np.abs(wa).max()
    U = random.uniform(0, 2 / b) * A.T

    for i in range(max_iterations):
        W = np.eye(n) - A @ U

        # if linalg.norm(W) > 1:
        #     raise ValueError('Алгоритм не сходится')

        if linalg.norm(W) < eps:
            break

        U = U @ (np.eye(n) + W)

    if verbose:
        print_U = tabulate(U, tablefmt="fancy_grid")
        print_E = tabulate(A @ U, tablefmt="fancy_grid")

        print('Матрица A^-1')
        print(print_U)
        print("Mатрица A * A^-1")
        print(print_E)

    return U


def spectral_radius(A, verbose=False, eps=1e-8, max_iterations=10000):
    """
    Итерационный метод поиска спектрального радиуса
    :param A: Квадратная матрица коэффициентов размером n на n
    :param b: Вектор-столбец размером n на 1
    :param verbose: Подробный вывод
    :param eps: Погрешность измерения
    :param max_iterations: Максимальное количество итераций
    :return: l - спектральный радиус, x - собственный вектор
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    # x0 = np.random.uniform(0, 1, size=(A.shape[0], 1))
    x0 = np.ones((A.shape[0], 1))
    x1 = np.zeros_like(x0)
    l0 = 0
    for i in range(max_iterations):
        x1 = A @ x0
        l1 = x1[0] / x0[0]
        if abs(l1 - l0) < eps:
            break
        x0 = x1
        l0 = l1

    x1 /= linalg.norm(x1)

    if verbose:
        print('Спектральный радиус', l1)
        x = tabulate(x1, tablefmt="fancy_grid")

        print('Собственный вектор')
        print(x)

    return l1, x1
