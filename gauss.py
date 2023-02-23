import numpy as np
from numpy import linalg
from solver import describe


def backward_lower(L, b):
    """
    Обратной ход метода Гаусса для нижней треугольной матрицы
    :param L: Треугольная матрица коэффициентов размером n на n
    :param b: Вектор-столбец размером n на 1
    :return: Вектор x - решение системы размером n на 1
    """
    n = L.shape[0]

    y = np.zeros((n, 1))
    for i in range(n):
        sum = 0
        for j in range(i):
            sum = sum + L[i, j] * y[j]
        y[i] = 1 / L[i, i] * (b[i] - sum)
    return y


def backward_upper(U, b):
    """
    Обратной ход метода Гаусса для верзней треугольной матрицы
    :param U: Треугольная матрица коэффициентов размером n на n
    :param b: Вектор-столбец размером n на 1
    :return: Вектор x - решение системы размером n на 1
    """
    n = U.shape[0]

    # Вектор X
    x = np.zeros((n, 1))

    # Обратный ход метода Гаусса
    for i in reversed(range(n)):
        sum = 0
        for j in range(i + 1, n):
            sum = sum + U[i, j] * x[j]
        x[i] = 1 / U[i, i] * (b[i] - sum)

    return x


def method_gauss_1(A, b, verbose=False):
    """
    Решение системы линейных алгебраических уравнений методом Гаусса
    (схема единственного деления)
    :param verbose: Вывести подробную информацию
    :param A: Квадратная матрица коэффициентов размером n на n
    :param b: Вектор-столбец размером n на 1
    :return: X - вектор-столбец решение системы размером n на 1
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    # Расширенная матрица
    Ab = np.hstack((A, b))
    n, m = Ab.shape

    # Прямой ход Гаусса
    for lineI in range(n):
        # Ведущий элемент
        leader = Ab[lineI, lineI]

        if leader == 0:
            raise ValueError('Получили 0 на диагонали. Система не может быть решена этим методом')

        for moveI in range(lineI + 1, n, 1):
            for moveJ in range(lineI + 1, m, 1):
                Ab[moveI, moveJ] = Ab[moveI, moveJ] - Ab[moveI, lineI] * Ab[lineI, moveJ] / leader

        # Двигаемся по строке и делим все элементы на ведущий
        for moveJ in range(lineI + 1, m, 1):
            Ab[lineI, moveJ] /= leader

        # Обнуляем все эелементы в столбце
        for moveI in range(lineI + 1, n, 1):
            Ab[moveI, lineI] = 0

        Ab[lineI, lineI] = 1

    # Обратный ход метода Гаусса
    new_A = Ab[:n, :n]
    new_b = Ab[:, -1]

    x = backward_upper(new_A, new_b)

    if verbose:
        describe(A, b, x)

    return x


def method_gauss_2(A, b, verbose=False):
    """
    Решение системы линейных алгебраических уравнений методом Гаусса
    (с выбором ведущего элемента)
    :param verbose: Вывести подробную информацию
    :param A: Квадратная матрица коэффициентов размером n на n
    :param b: Вектор-столбец размером n на 1
    :return: Решение системы линейный уравнений (вектор-столбец X размером n на 1)
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError('Неправильный размер матрицы A')

    if linalg.det(A) == 0:
        raise ValueError('Определитель матрицы равен 0 (detA = 0)')

    # Расширенная матрица
    Ab = np.hstack((A, b))
    n, m = Ab.shape

    # Прямой ход Гаусса
    for lineI in range(n):

        # Переставляем строки так, чтобы наибольший по модулю коэффициент при переменной x попал на главную диагональ
        find_flag = False
        index_max_line = 0
        max_element = Ab[lineI, lineI]
        for moveI in range(lineI + 1, n):
            if abs(max_element) < abs(Ab[moveI, lineI]):
                max_element = Ab[moveI, lineI]
                index_max_line = moveI
                find_flag = True

        # Меняем местами строки
        if find_flag:
            Ab[[lineI, index_max_line]] = Ab[[index_max_line, lineI]]

        # Ведущий элемент
        leader = Ab[lineI, lineI]

        if leader == 0:
            raise ValueError('Получили 0 на диагонали. Система не может быть решена этим методом')

        for moveI in range(lineI + 1, n, 1):
            for moveJ in range(lineI + 1, m, 1):
                Ab[moveI, moveJ] = Ab[moveI, moveJ] - Ab[moveI, lineI] * Ab[lineI, moveJ] / leader

        # Двигаемся по строке и делим все элементы на ведущий
        for moveJ in range(lineI + 1, m, 1):
            Ab[lineI, moveJ] /= leader

        # Обнуляем все эелементы в столбце
        for moveI in range(lineI + 1, n, 1):
            Ab[moveI, lineI] = 0

        Ab[lineI, lineI] = 1

    # Обратный ход метода Гаусса
    new_A = Ab[:n, :n]
    new_b = Ab[:, -1]

    x = backward_upper(new_A, new_b)

    if verbose:
        describe(A, b, x)

    return x
