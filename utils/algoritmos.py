import math
import sys


def bissecao(f, a, b, TOL, iter=100):
    if TOL <= 0:
        raise ValueError("A tolerância deve ser positiva.")
    maximum = int(iter)
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if fa*fb > 0:
        raise ValueError("Nenhuma raiz encontrada no intervalo.")

    midpoint = (a+b)/2
    for count in range(1, maximum+1):
        midpoint = (a+b)/2
        fm = f(midpoint)
        if fm == 0 or abs(b-a)/2 <= TOL:
            return midpoint, count
        if fa*fm < 0:
            b, fb = midpoint, fm
        else:
            a, fa = midpoint, fm
    return midpoint, maximum


def pontofixo(a, g, TOL=1e-8, iter=1000):
    current = float(a)
    for _ in range(int(iter)):
        following = float(g(current))
        if not math.isfinite(following):
            raise ValueError("A iteração produziu um valor não finito.")
        if abs(following-current) <= TOL:
            return following
        current = following
    raise RuntimeError("O método do ponto fixo não convergiu.")


def newton_raphson(a, f, TOL=1e-8, df=None, iter=100):
    def numerical_derivative(x):
        step = math.sqrt(sys.float_info.epsilon)*max(1.0, abs(x))
        return (f(x+step)-f(x-step))/(2*step)

    derivative = df or numerical_derivative
    return pontofixo(a, lambda x: x-f(x)/derivative(x), TOL, iter)


def secante(a, b, f, TOL=1e-8, iter=100):
    previous, current = float(a), float(b)
    f_previous, f_current = f(previous), f(current)
    for _ in range(int(iter)):
        difference = f_current-f_previous
        if abs(difference) <= sys.float_info.epsilon:
            raise ZeroDivisionError("Denominador nulo no método da secante.")
        following = current-f_current*(current-previous)/difference
        f_following = f(following)
        if abs(following-current) <= TOL or abs(f_following) <= TOL:
            return following
        previous, current = current, following
        f_previous, f_current = f_current, f_following
    raise RuntimeError("O método da secante não convergiu.")
