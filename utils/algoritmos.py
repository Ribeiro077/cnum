import math
import sys
import numpy as np


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


def _sistema(A, B):
    """Converte e verifica a matriz quadrada e o vetor independente."""
    matriz = np.asarray(A, dtype=float)
    vetor = np.asarray(B, dtype=float)
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("A deve ser uma matriz quadrada.")
    if vetor.shape != (matriz.shape[0],):
        raise ValueError("B deve ter um elemento por linha de A.")
    if not np.all(np.isfinite(matriz)) or not np.all(np.isfinite(vetor)):
        raise ValueError("O sistema deve conter apenas valores finitos.")
    return matriz, vetor


def lu_pivot(A: np.ndarray):
    """Fatora A com pivoteamento parcial, retornando P, L, U (P@A=L@U)."""
    U = np.asarray(A, dtype=float).copy()
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("A deve ser uma matriz quadrada.")
    if not np.all(np.isfinite(U)):
        raise ValueError("A deve conter apenas valores finitos.")

    n = U.shape[0]
    P = np.eye(n)
    L = np.eye(n)
    escala = np.max(np.abs(U)) if U.size else 0.0
    limite = np.finfo(float).eps * max(1, n) * escala

    for coluna in range(n):
        pivo = coluna + np.argmax(np.abs(U[coluna:, coluna]))
        if abs(U[pivo, coluna]) <= limite:
            raise np.linalg.LinAlgError("Matriz singular; não existe solução única.")

        if pivo != coluna:
            U[[coluna, pivo], :] = U[[pivo, coluna], :]
            P[[coluna, pivo], :] = P[[pivo, coluna], :]
            L[[coluna, pivo], :coluna] = L[[pivo, coluna], :coluna]

        for linha in range(coluna + 1, n):
            fator = U[linha, coluna] / U[coluna, coluna]
            L[linha, coluna] = fator
            U[linha, coluna:] -= fator * U[coluna, coluna:]
            U[linha, coluna] = 0.0

    return P, L, U


def lb(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Substituição progressiva para L y = B."""
    L, B = _sistema(L, B)
    y = np.empty_like(B)
    for i in range(len(B)):
        if L[i, i] == 0:
            raise np.linalg.LinAlgError("Diagonal nula em L.")
        y[i] = (B[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def uy(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Substituição regressiva para U x = Y."""
    U, Y = _sistema(U, Y)
    x = np.empty_like(Y)
    for i in range(len(Y) - 1, -1, -1):
        if U[i, i] == 0:
            raise np.linalg.LinAlgError("Diagonal nula em U.")
        x[i] = (Y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def lu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Resolve A x = B por fatoração LU com pivoteamento parcial."""
    A, B = _sistema(A, B)
    P, L, U = lu_pivot(A)
    return uy(U, lb(L, P @ B))


def _iteracoes(A, B, k, TOL):
    A, B = _sistema(A, B)
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k deve ser um número inteiro positivo.")
    if not np.isfinite(TOL) or TOL <= 0:
        raise ValueError("TOL deve ser positiva e finita.")
    if np.any(np.diag(A) == 0):
        raise ValueError("A possui elemento diagonal nulo; permute as linhas.")
    return A, B


def jacobi(A: np.ndarray, B: np.ndarray, k: int, TOL: float) -> np.ndarray:
    """Itera desde x0=0 até a tolerância ou até completar k passos."""
    A, B = _iteracoes(A, B, k, TOL)
    diagonal = np.diag(A)
    restante = A - np.diag(diagonal)
    x = np.zeros(len(B))
    for _ in range(k):
        proximo = (B - restante @ x) / diagonal
        if np.linalg.norm(proximo - x) < TOL:
            return proximo
        x = proximo
    return x


def seidel(A: np.ndarray, B: np.ndarray, k: int, TOL: float) -> np.ndarray:
    """Gauss–Seidel desde x0=0, usando componentes recém-atualizadas."""
    A, B = _iteracoes(A, B, k, TOL)
    x = np.zeros(len(B))
    for _ in range(k):
        anterior = x.copy()
        for i in range(len(B)):
            esquerda = A[i, :i] @ x[:i]
            direita = A[i, i + 1:] @ anterior[i + 1:]
            x[i] = (B[i] - esquerda - direita) / A[i, i]
        if np.linalg.norm(x - anterior) < TOL:
            return x
    return x
