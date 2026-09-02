import math
import sys

def bissecao(f, a, b, TOL, iter=100):
    """Encontra uma raiz em [a, b] pelo método da bisseção."""
    if TOL <= 0:
        raise ValueError("A tolerância deve ser positiva.")
    max_iter = int(iter)
    if max_iter <= 0:
        raise ValueError("O número máximo de iterações deve ser positivo.")

    fa = f(a)
    fb = f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if fa * fb > 0:
        raise ValueError("Nenhuma raiz encontrada no intervalo.")

    c = (a + b) / 2.0
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        if fc == 0 or abs(b - a) / 2.0 <= TOL:
            return c, i
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return c, max_iter

def pontofixo(a, g, TOL=1e-8, iter=1000):
    """Executa x_(n+1)=g(x_n) até estabilizar."""
    if TOL <= 0:
        raise ValueError("A tolerância deve ser positiva.")

    x = float(a)
    for i in range(1, int(iter) + 1):
        next_x = float(g(x))
        if not math.isfinite(next_x):
            raise ValueError("A iteração produziu um valor não finito.")
        if abs(next_x - x) <= TOL * max(1.0, abs(next_x)):
            return next_x, i
        x = next_x
    raise RuntimeError("O método do ponto fixo não convergiu.")

def newton_raphson(a, f, TOL=1e-8, df=None, iter=100):
    """Método de Newton-Raphson com derivada analítica ou diferença central."""
    def numerical_derivative(x):
        h = math.sqrt(sys.float_info.epsilon) * max(1.0, abs(x))
        return (f(x + h) - f(x - h)) / (2.0 * h)

    derivative = df or numerical_derivative

    def iteration(x):
        slope = derivative(x)
        if abs(slope) <= sys.float_info.epsilon:
            raise ZeroDivisionError("Derivada nula ou muito próxima de zero.")
        return x - f(x) / slope

    return pontofixo(a, iteration, TOL, iter)

def secante(a, b, f, TOL=1e-8, iter=100):
    """Método da secante com limite de iterações e teste de denominador."""
    x0, x1 = float(a), float(b)
    f0, f1 = f(x0), f(x1)

    for i in range(1, int(iter) + 1):
        denominator = f1 - f0
        if abs(denominator) <= sys.float_info.epsilon:
            raise ZeroDivisionError("Denominador nulo ou muito próximo de zero.")
        x2 = x1 - f1 * (x1 - x0) / denominator
        if not math.isfinite(x2):
            raise ValueError("A iteração produziu um valor não finito.")
        f2 = f(x2)
        if abs(f2) <= TOL:
            return x2, i
        if abs(x2 - x1) <= TOL * max(1.0, abs(x2)):
            raise RuntimeError("O método da secante estagnou sem atingir a tolerância.")
        x0, x1 = x1, x2
        f0, f1 = f1, f2

    raise RuntimeError("O método da secante não convergiu.")
