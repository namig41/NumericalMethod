import numpy as np
import gauss
import iteralg

from decomposition import get_LU


def main():
    A = np.array([[10, 1, 1], [2, 10, 1], [2, 2, 10]], dtype=float)
    b = np.array([[12, 13, 14]], dtype=float)
    # x = gauss.method_gauss_1(A, b.T)
    x = iteralg.method_seidel_normal(A, b.T, verbose=True)


if __name__ == '__main__':
    main()
