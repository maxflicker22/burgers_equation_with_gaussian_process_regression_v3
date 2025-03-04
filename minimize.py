import numpy as np
def unwrap(s):
    """Extract the numerical values from `s` into a column vector `v`."""
    if isinstance(s, np.ndarray):
        return s.flatten()
    elif isinstance(s, dict):
        v = []
        for key in sorted(s.keys()):
            v.extend(unwrap(s[key]))
        return np.array(v)
    elif isinstance(s, list):
        v = []
        for item in s:
            v.extend(unwrap(item))
        return np.array(v)
    else:
        return []

def rewrap(s, v):
    """Map the numerical elements in the vector `v` onto the variables `s`."""
    if isinstance(s, np.ndarray):
        size = s.size
        return v[:size].reshape(s.shape), v[size:]
    elif isinstance(s, dict):
        for key in sorted(s.keys()):
            s[key], v = rewrap(s[key], v)
        return s, v
    elif isinstance(s, list):
        for i in range(len(s)):
            s[i], v = rewrap(s[i], v)
        return s, v
    else:
        return s, v

def feval(f, x, *args):
    """Evaluate the function `f` and its gradient at `x`."""
    print(f"returned feval {f(x, *args)}")
    return f(x, *args)


def minimize(X, f, length, *args):
    INT = 0.1
    EXT = 3.0
    MAX = 100
    RATIO = 10
    SIG = 0.1
    RHO = SIG / 2

    if isinstance(length, (list, tuple)) and len(length) == 2:
        red = length[1]
        length = length[0]
    else:
        red = 1

    if length > 0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation'

    i = 0
    ls_failed = 0

    f0, df0 = feval(f, X, *args)
    Z = X
    X = unwrap(X)
    df0 = unwrap(df0)
#    print(f'{S} {i:6d};  Value {f0:4.6e}\r')

    fX = [f0]
    i += (length < 0)
    s = -(1)*df0
    d0 = -np.dot(s, s)
    x3 = red / (1 - d0)

    while i < abs(length):
        i += (length > 0)

        X0 = X.copy()
        F0 = f0
        dF0 = df0.copy()
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)

        while True:
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0.copy()
            success = False
            while not success and M > 0:
                try:
                    M -= 1
                    i += (length < 0)
                    f3, df3 = feval(f, rewrap(Z, X + x3 * s), *args)
                    df3 = unwrap(df3)
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3) + np.isinf(df3)):
                        raise ValueError()
                    success = True
                except:
                    x3 = (x2 + x3) / 2

            if f3 < F0:
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3.copy()

            d3 = np.dot(df3, s)
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
                break

            x1 = x2
            f1 = f2
            d1 = d2
            x2 = x3
            f2 = f3
            d2 = d3
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            x3 = x1 - d1 * (x2 - x1) ** 2 / (B + np.sqrt(B * B - A * d1 * (x2 - x1)))
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:
                x3 = x2 * EXT
            elif x3 > x2 * EXT:
                x3 = x2 * EXT
            elif x3 < x2 + INT * (x2 - x1):
                x3 = x2 + INT * (x2 - x1)

        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:
                x4 = x3
                f4 = f3
                d4 = d3
            else:
                x2 = x3
                f2 = f3
                d2 = d3
            if f4 > f0:
                x3 = x2 - (0.5 * d2 * (x4 - x2) ** 2) / (f4 - f2 - d2 * (x4 - x2))
            else:
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A
            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2
            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))

            f3, df3 = feval(f, rewrap(Z, X + x3 * s), *args)
            df3 = unwrap(df3)
            if f3 < F0:
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3.copy()

            M -= 1
            i += (length < 0)
            d3 = np.dot(df3, s)

        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:
            X = X + x3 * s
            f0 = f3
            fX.append(f0)
            print(f'{S} {i:6d};  Value {f0:4.6e}\r')
            s = (np.dot(df3, df3) - np.dot(df0, df3)) / np.dot(df0, df0) * s - df3
            df0 = df3.copy()
            d3 = d0
            d0 = np.dot(df0, s)
            if d0 > 0:
                s = -df0
                d0 = -np.dot(s, s)
            x3 = x3 * min(RATIO, d3 / (d0 - np.finfo(float).eps))
            ls_failed = 0
        else:
            X = X0
            f0 = F0
            df0 = dF0.copy()
            if ls_failed or i > abs(length):
                break
            s = -df0
            d0 = -np.dot(s, s)
            x3 = 1 / (1 - d0)
            ls_failed = 1

    X = rewrap(Z, X)[0]

    print()
    return X, fX, i
