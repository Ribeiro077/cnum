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
