import numpy as np
import gauss


def main():
    A = np.array([[2, 1, 4], [3, 2, 1], [1, 3, 3]], dtype=float)
    b = np.array([[16, 10, 16]], dtype=float)
    x = gauss.method_gauss_1(A, b.T)
    print(x)


if __name__ == '__main__':
    main()
