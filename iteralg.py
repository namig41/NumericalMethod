import numpy as np
from numpy import linalg
from solver import describe


def method_simple_iterations(A, b, verbose=False, eps=1e-8, max_iterations=10000):
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
