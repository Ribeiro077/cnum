def bissecao(f, a, b, TOL, iter=100):
    """Encontra uma raiz de f no intervalo [a, b] pelo método da bisseção."""
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Nenhuma raiz encontrada no intervalo.")

    c = (a + b) / 2.0
    i = 0
    erro = abs(b - a)

    while erro > TOL and i < iter:
        c = (a + b) / 2.0
        fc = f(c)

        if fc == 0:
            return c, i
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        i += 1
        erro = abs(b - a)

    return c, i