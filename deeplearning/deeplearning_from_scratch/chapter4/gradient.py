import numpy as np


def numerical_gradient(f, x):
    # 0.0001
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 값 복원
        x[idx] = tmp_val
        it.iternext()

    return grad
