import numpy as np

############ LABO 0 ############

def esCuadrada(A):
    cant_filas, cant_columnas = A.shape
    return cant_filas == cant_columnas


def diagonal(A):
    cant_filas, cant_columnas = A.shape

    D = A.copy()
    for i in range(cant_filas):
        for j in range(cant_columnas):
            if i != j:
                D[i][j] = 0
    return D


def traza(A):
    suma = 0
    for i in range(len(A)):
        suma += A[i][i]
    return suma


def traspuesta(A):
    At = []
    cant_filas, cant_columnas = A.shape

    for i in range(cant_columnas):
        nueva_fila = []

        for j in range(cant_filas):
            nueva_fila.append(A[j][i])

        At.append(nueva_fila)

    return np.array(At)


def esSimetrica(A):
    cant_filas, cant_columnas = A.shape

    for i in range(cant_filas):
        for j in range(cant_columnas):
            if i != j:
                if A[i][j] != A[j][i]:
                    return False

    return True

############### LABO 1 #################

def error(x, y):
    return abs(x - np.float64(y))


def error_relativo(x, y):
    return error(x, y) / abs(x)


def matricesIguales(A, B):
    f, c = A.shape
    if ((f, c) != B.shape):
        return False

    eps = np.finfo('float').eps

    for i in range(f):
        for j in range(c):
            if error(A[i, j], B[i, j]) > eps:
                return False

    return True


################ LABO 2 #################

def rota(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])


def escala(s):
    n = len(s)
    matriz = np.zeros((n, n))
    for i in range(n):
        matriz[i][i] = s[i]
 
    return matriz


def rota_y_escala(theta, s):
    return escala(s) @ rota(theta)


def afin(theta, s, b):
    T_lineal = rota_y_escala(theta, s)

    T = np.eye(3)

    T[:2, :2] = T_lineal
    T[:2, 2] = b
    return T
                 

def trans_afin(v, theta, s, b):
    T = afin(theta, s, b)

    v_hom = np.array([v[0], v[1], 1])
    w_hom = T @ v_hom
    
    return w_hom[:2]


################# LABO 4 #################

def calculaLU(A):
    if A is None:
        return None, None, 0

    try:
        U = np.array(A, dtype = float)
    except:
        return None, None, 0

    if np.ndim(U) != 2 or np.shape(U)[0] != np.shape(U)[1]:
        return None, None, 0

    n = np.shape(U)[0]
    L = np.eye(n)
    nops = 0

    for i in range(n - 1):
        pivot = U[i][i]
        
        if pivot == 0:
            return None, None, 0

        for j in range(i + 1, n):
            coef = U[j][i] / pivot
            nops += 1
            
            L[j][i] = coef
            U[j][i] = 0
            
            U[j][i + 1:] = U[j][i + 1:] - coef * U[i][i + 1:]
            
            nops += 2 * (n - i - 1)

    return L, U, nops


def res_tri(A, b, inferior=True):
    """
    Resuelve Ax = b cuando A es triangular inferior o superior.
    Retorna None si las dimensiones no son compatibles o A es singular.
    """
    try:
        T = np.array(A, dtype=float)
        termino_independiente = np.array(b, dtype=float)
    except:
        return None

    if np.ndim(T) != 2 or np.shape(T)[0] != np.shape(T)[1]:
        return None
    if np.ndim(termino_independiente) != 1:
        return None

    n = np.shape(T)[0]
    if np.shape(termino_independiente)[0] != n:
        return None

    x = np.zeros(n)

    if inferior:
        for i in range(n):
            if T[i][i] == 0:
                return None

            suma = 0.0
            for j in range(i):
                suma += T[i][j] * x[j]
            x[i] = (termino_independiente[i] - suma) / T[i][i]
    else:
        for i in range(n - 1, -1, -1):
            if T[i][i] == 0:
                return None

            suma = 0.0
            for j in range(i + 1, n):
                suma += T[i][j] * x[j]
            x[i] = (termino_independiente[i] - suma) / T[i][i]

    return x


def inversa(A):
    """
    Calcula la inversa de A mediante su factorizacion LU.
    """
    resultado_lu = calculaLU(A)
    L, U, _ = resultado_lu

    if L is None or U is None:
        return None

    n = np.shape(U)[0]
    A_inversa = np.zeros((n, n))

    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0

        y = res_tri(L, e_j)
        if y is None:
            return None

        x = res_tri(U, y, inferior=False)
        if x is None:
            return None

        A_inversa[:, j] = x

    return A_inversa


def calculaLDV(A):
    """
    Calcula A = LDV a partir de la factorizacion LU de A.
    """
    L, U, _ = calculaLU(A)

    if L is None or U is None:
        return None

    n = np.shape(U)[0]
    D = np.zeros((n, n))
    V = np.zeros((n, n))

    for i in range(n):
        diagonal = U[i][i]
        if diagonal == 0:
            return None

        D[i][i] = diagonal
        V[i][i:] = U[i][i:] / diagonal

    return L, D, V


def esSDP(A, atol=1e-10):
    """
    Decide si A es simetrica definida positiva mediante su LDV.
    """
    if atol < 0:
        return False

    try:
        matriz = np.array(A, dtype=float)
    except:
        return False

    if np.ndim(matriz) != 2 or np.shape(matriz)[0] != np.shape(matriz)[1]:
        return False

    n = np.shape(matriz)[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matriz[i][j] - matriz[j][i]) > atol:
                return False

    resultado_ldv = calculaLDV(matriz)
    if resultado_ldv is None:
        return False

    _, D, _ = resultado_ldv
    for i in range(n):
        if D[i][i] <= atol:
            return False

    return True


def calculaCholesky(A, atol=1e-10):
    """
    Calcula A = R R^t, con R triangular inferior, cuando A es SDP.
    """
    if not esSDP(A, atol):
        return None

    L, D, _ = calculaLDV(A)
    n = np.shape(L)[0]
    R = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            R[i][j] = L[i][j] * D[j][j] ** 0.5

    return R


