import numpy as np
from numpy import linalg
import math


def vnorm_1(vec):
    """
    :param vec: Вектор
    :return: Первая векторная норма — это максимум среди модулей элементов столбца

    """

    return np.abs(vec).max()


def vnorm_2(vec):
    """
    :param vec: Вектор
    :return: Вторая векторная норма — сумма модулей элементов столбца
    """

    return np.abs(vec).sum()


def vnorm_3(vec):
    """
    :param vec: Вектор
    :return: Третья векторная норма — евклидова норма, т.е. квадратный корень из суммы квадратов элементов
    """

    return math.sqrt(np.power(vec, 2).sum())


def mnorm_1(matrix):
    """
    :param matrix: Матрица
    :return: Первая матричная норма — это максимум суммы модулей элементов в строке
    """

    max_sum = 0
    for row in matrix:
        sum = 0
        for col in row:
            sum += abs(col)
        if sum > max_sum:
            max_sum = sum
    return max_sum


def mnorm_2(matrix):
    """
    :param matrix: Матрица
    :return: Вторая матричная норма — это максимум суммы модулей элементов в столбце;
    """

    return mnorm_1(matrix.T)


def mnorm_4(matrix):
    """

    :param matrix: Матрица
    :return: Четвертая матричная норма — это квадратный корень из суммы квадратов элементов

    """
    sum = 0
    for row in matrix:
        for col in row:
            sum += col ** 2
    return math.sqrt(sum)


def cond(matrix):
    """

    :param matrix: Матрица
    :return: Число обусловленности - степень чувствительности системы к входным данным. Чем больше это число, тем хуже обусловленность системы
    """
    return mnorm_1(matrix) * mnorm_1(linalg.inv(matrix))
