def numerical_diff(f, x):
    # 0.0001
    h = 1e-4
    return (f(x + h) - f(x-h)) / (2 * h)

