import numpy as np


def copiar_matriz(A):
    if len(A.shape) == 1:
        copia = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            copia[i] = A[i]
        return copia

    filas, columnas = A.shape
    copia = np.zeros((filas, columnas))
    for i in range(filas):
        for j in range(columnas):
            copia[i, j] = A[i, j]

    return copia


def esCuadrada(A):
    """Indica si A es cuadrada.

    Parámetros:
        A: matriz representada como un array bidimensional.
    Retorna:
        True si A tiene igual cantidad de filas y columnas; False en caso contrario.
    """
    filas, columnas = A.shape
    return filas == columnas


def diagonal(A):
    """Construye la matriz diagonal asociada a A.

    Parámetros:
        A: matriz representada como un array bidimensional.
    Retorna:
        Una matriz con la diagonal de A y ceros en las demás posiciones.
    """
    filas, columnas = A.shape

    D = np.zeros((filas, columnas))

    for i in range(min(filas, columnas)):
        D[i, i] = A[i, i]

    return D


def traza(A):
    """Calcula la traza de una matriz cuadrada.

    Parámetros:
        A: matriz cuadrada.
    Retorna:
        La suma de los elementos de la diagonal principal.
    """
    suma = 0
    for i in range(len(A)):
        suma += A[i, i]
    return suma


def traspuesta(A):
    """Calcula la traspuesta de A.

    Parámetros:
        A: matriz representada como un array bidimensional.
    Retorna:
        Un array con filas y columnas intercambiadas.
    """
    At = []
    filas, columnas = A.shape

    for i in range(columnas):
        nueva_fila = []

        for j in range(filas):
            nueva_fila.append(A[j, i])

        At.append(nueva_fila)

    return np.array(At)


def esSimetrica(A):
    """Indica si A es simétrica.

    Parámetros:
        A: matriz cuadrada representada como un array bidimensional.
    Retorna:
        True si A coincide con su traspuesta; False en caso contrario.
    """
    filas, columnas = A.shape

    for i in range(filas):
        for j in range(columnas):
            if i != j:
                if A[i, j] != A[j, i]:
                    return False

    return True


def error(x, y):
    """Calcula el error absoluto entre x e y.

    Parámetros:
        x: primer valor de la comparación.
        y: segundo valor de la comparación.
    Retorna:
        El valor absoluto de la diferencia entre x e y.
    """
    # TODO: NO PERMITIDO: np.float64 no figura entre los tipos habilitados.
    return abs(x - np.float64(y))


def error_relativo(x, y):
    """Calcula el error relativo entre x e y.

    Parámetros:
        x: valor de referencia, distinto de cero.
        y: valor que se compara con x.
    Retorna:
        El error absoluto entre ambos valores dividido por el valor absoluto de x.
    """
    return error(x, y) / abs(x)


def matricesIguales(A, B):
    """Compara dos matrices elemento a elemento con tolerancia de máquina.

    Parámetros:
        A: primera matriz.
        B: segunda matriz.
    Retorna:
        True si tienen igual forma y sus elementos difieren a lo sumo en eps;
        False en caso contrario.
    """
    f, c = A.shape
    if ((f, c) != B.shape):
        return False

    # TODO: NO PERMITIDO: np.finfo no figura entre las funciones habilitadas.
    eps = np.finfo('float').eps

    for i in range(f):
        for j in range(c):
            if error(A[i, j], B[i, j]) > eps:
                return False

    return True


def rota(theta):
    """Construye la matriz de rotación para un ángulo theta.

    Parámetros:
        theta: ángulo de rotación en radianes.
    Retorna:
        La matriz de rotación bidimensional asociada a theta.
    """
    # TODO: NO PERMITIDO: np.cos y np.sin no figuran entre las funciones habilitadas.
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def escala(s):
    """Construye una matriz diagonal de escalamiento.

    Parámetros:
        s: vector con los factores de escala.
    Retorna:
        Una matriz diagonal cuyos elementos diagonales son los de s.
    """
    n = len(s)
    matriz = np.zeros((n, n))
    for i in range(n):
        matriz[i, i] = s[i]

    return matriz


def rota_y_escala(theta, s):
    """Combina el escalamiento indicado con una rotación.

    Parámetros:
        theta: ángulo de rotación en radianes.
        s: vector con los factores de escala.
    Retorna:
        La matriz que resulta de aplicar la rotación y el escalamiento.
    """
    # TODO: NO PERMITIDO: el producto matricial debe implementarse por el grupo.
    return escala(s) @ rota(theta)


def afin(theta, s, b):
    """Construye una transformación afín bidimensional homogénea.

    Parámetros:
        theta: ángulo de rotación en radianes.
        s: vector con los factores de escala.
        b: vector de traslación bidimensional.
    Retorna:
        La matriz homogénea 3x3 de la transformación afín.
    """
    T_lineal = rota_y_escala(theta, s)

    T = np.eye(3)

    T[:2, :2] = T_lineal
    T[:2, 2] = b
    return T


def trans_afin(v, theta, s, b):
    """Aplica una transformación afín a un punto bidimensional.

    Parámetros:
        v: punto de entrada con dos coordenadas.
        theta: ángulo de rotación en radianes.
        s: vector con los factores de escala.
        b: vector de traslación bidimensional.
    Retorna:
        El punto transformado, como vector de dos coordenadas.
    """
    T = afin(theta, s, b)

    v_hom = np.array([v[0], v[1], 1])
    # TODO: NO PERMITIDO: el producto matricial debe implementarse por el grupo.
    w_hom = T @ v_hom

    return w_hom[:2]


def norma(x, p):
    if p == 1:
        return sum([abs(n) for n in x])

    elif p == "inf":
        max = 0
        for n in x:
            if abs(n) > max:
                max = abs(n)
        return max

    suma = sum([n ** p for n in x])
    return np.power(suma, 1 / p)


def normaliza(X, p):
    return [x / norma(x, p) for x in X]


# def normaMatMC(A, q, p, Np):
#     """
#     Devuelve la norma A-{q,p} y el vector x en el cual se alcanza
#     el maximo.
#     """
#     vectoresAlAzar = np.random.rand(Np, A.shape[1])
#     vectoresNormalizados = normaliza(vectoresAlAzar, p)
#     vectorConNorma = map(lambda x: [norma(A@x, q), x], vectoresNormalizados)
#     return max(vectorConNorma, key=lambda p: p[0])

def normaExacta(A, p=[1, "inf"]):
    def normaM(A, p):
        if p == "inf":
            sumas_inf = np.zeros(A.shape[0])
            for i in range(0, A.shape[0]):
                for j in range(0, A.shape[1]):
                    sumas_inf[i] += abs(A[i][j])
            return max(sumas_inf)

        if p == 1:
            sumas_1 = np.zeros(A.shape[1])
            for j in range(0, A.shape[1]):
                for i in range(0, A.shape[0]):
                    sumas_1[j] += abs(A[i][j])
            return max(sumas_1)

    if p == "inf": return normaM(A, "inf")

    if isinstance(p, int): return normaM(A, p)

    if p == [1, "inf"]: return [normaM(A, 1), normaM(A, "inf")]

    return None


def condMC(A, p):
    ...


def condExacto(A, p):
    ...


################# LABO 4 #################

def calculaLU(A):
    """Calcula la factorización LU de A sin pivoteo.

    Parámetros:
        A: matriz cuadrada a factorizar.
    Retorna:
        La tupla (L, U, nops), con L triangular inferior, U triangular superior
        y la cantidad de operaciones; si la entrada no es válida o no se puede
        factorizar, retorna (None, None, 0).
    """
    if A is None:
        return None, None, 0

    try:
        U = np.array(A, dtype=float)
    except:
        return None, None, 0

    # TODO: NO PERMITIDO: np.ndim no figura entre las funciones habilitadas.
    if np.ndim(U) != 2 or U.shape[0] != U.shape[1]:
        return None, None, 0

    n = U.shape[0]
    L = np.eye(n)
    nops = 0

    for i in range(n - 1):
        pivot = U[i, i]

        if pivot == 0:
            return None, None, 0

        for j in range(i + 1, n):
            coef = U[j, i] / pivot
            nops += 1

            L[j, i] = coef
            U[j, i] = 0

            U[j, i + 1:] = U[j, i + 1:] - coef * U[i, i + 1:]

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

    # TODO: NO PERMITIDO: np.ndim no figura entre las funciones habilitadas.
    if np.ndim(T) != 2 or T.shape[0] != T.shape[1]:
        return None
    # TODO: NO PERMITIDO: np.ndim no figura entre las funciones habilitadas.
    if np.ndim(termino_independiente) != 1:
        return None

    n = T.shape[0]
    if termino_independiente.shape[0] != n:
        return None

    x = np.zeros(n)

    if inferior:
        for i in range(n):
            if T[i, i] == 0:
                return None

            suma = 0.0
            for j in range(i):
                suma += T[i, j] * x[j]
            x[i] = (termino_independiente[i] - suma) / T[i, i]
    else:
        for i in range(n - 1, -1, -1):
            if T[i, i] == 0:
                return None

            suma = 0.0
            for j in range(i + 1, n):
                suma += T[i, j] * x[j]
            x[i] = (termino_independiente[i] - suma) / T[i, i]

    return x


def inversa(A):
    """
    Calcula la inversa de A mediante su factorizacion LU.
    """
    resultado_lu = calculaLU(A)
    L, U, _ = resultado_lu

    if L is None or U is None:
        return None

    n = U.shape[0]
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

    n = U.shape[0]
    D = np.zeros((n, n))
    V = np.zeros((n, n))

    for i in range(n):
        diagonal = U[i, i]
        if diagonal == 0:
            return None

        D[i, i] = diagonal
        V[i, i:] = U[i, i:] / diagonal

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

    # TODO: NO PERMITIDO: np.ndim no figura entre las funciones habilitadas.
    if np.ndim(matriz) != 2 or matriz.shape[0] != matriz.shape[1]:
        return False

    n = matriz.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matriz[i, j] - matriz[j, i]) > atol:
                return False

    resultado_ldv = calculaLDV(matriz)
    if resultado_ldv is None:
        return False

    _, D, _ = resultado_ldv
    for i in range(n):
        if D[i, i] <= atol:
            return False

    return True


def calculaCholesky(A, atol=1e-10):
    """
    Calcula A = R R^t, con R triangular inferior, cuando A es SDP.
    """
    if not esSDP(A, atol):
        return None

    L, D, _ = calculaLDV(A)
    n = L.shape[0]
    R = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            R[i, j] = L[i, j] * D[j, j] ** 0.5

    return R


################# LABO 5 #################

def GS(A, tol, retorna_nops):
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)

    n = len(A)

    r = 0
    norma_a0 = norma(A[:, 0])
    if norma_a0 > tol:
        Q[:, 0] = A[:, 0] / norma_a0
        R[0, 0] = norma_a0
        nops = 2 + (n + (n - 1) + 1)
        r += 1

    for j in range(1, n):
        q_j_moño = copiar_matriz(A[:, j])

        for k in range(0, j):
            R[k, j] = Q[:, k].T @ q_j_moño
            q_j_moño = q_j_moño - R[k, j] * Q[:, k]
            nops += (2 * n - 1) + 2

        norma_qj_moño = norma(q_j_moño)

        if norma_qj_moño > tol:
            R[r, j] = norma_qj_moño
            Q[:, r] = q_j_moño / R[r, j]
            nops += 2 + (n + (n - 1) + 1) + n
            r += 1

    Q_final = Q[:, :r]
    R_final = R[:r, :]

    if retorna_nops:
        return Q, R, nops

    return Q, R


### Funciones L05-QR
def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    if A is None: return None
    if len(A) != len(A[0]): return None

    return GS(A, tol, retorna_nops)


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve: 
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso


    """
    filas, columnas = A.shape
    if filas < columnas:
        return None

    R = copiar_matriz(A)

    Q = np.eye(filas)

    extra_info = {
        'R_matrices': [],
        'Q_matrices': []
    }

    for k in range(columnas):
        if extras:
            extra_info['R_matrices'].append(copiar_matriz(R))
            extra_info['Q_matrices'].append(copiar_matriz(Q))

        # tajada vertical de la submatriz abajo a la derecha a partir de R[k,k]
        x = R[k:, k]

        # norma con - evita cancelacion numerica
        alpha = -signo(x[0]) * norma(x, 2)

        # definimos u perpendicular al espejo
        # columna x -  normaSignoAlternado* primer canonico con dimension (m-k+1)
        u = x - alpha * np.eye(filas - k)[0]
        norma_u = norma(u, 2)
        if norma_u > tol:  # si no lo es, no lo procesamos, ya habia ceros debajo de la diagonal, pasamos a sig columna
            # si norma mayor a cero, reflejamos
            # normalizo
            u = u / norma_u
            # submatriz de reflectores espejo, dimension (m-k)x(m-k)
            H_k = np.eye(filas - k) - 2 * multiplacionMatricialDeVectores(u, u)
            # expandimos a la dimension de la matriz original, agrgando identidad arriba a la izquierda
            H_kp = np.eye(filas)
            H_kp[k:filas, k:filas] = H_k
            R = multiplicar(H_kp, R)
            Q = multiplicar(Q, H_kp)

    if extras:
        extra_info['R_matrices'].append(copiar_matriz(R))
        extra_info['Q_matrices'].append(copiar_matriz(Q))
        return Q, R, extra_info

    return Q, R


def calculaQR(A, metodo="RH", tol=1e-12):
    if metodo == "GS":
        return QR_con_GS(A, tol)
    elif metodo == "RH":
        return QR_con_HH(A, tol)
    else:
        return None


################# LABO 6 #################

def multMat(A, B):
    filasA = A.shape[0]
    columnasB = B.shape[1]
    columnasA = A.shape[1]
    AB = np.zeros((filasA, columnasB))

    for i in range(filasA):
        for j in range(columnasB):
            for k in range(columnasA):
                AB[i, j] += A[i, k] * B[k, j]

    return AB


def metpot2k(A, tol=1e-15, K=1000):
    def f(A, v):
        res = multMat(A, v)
        return res / norma(res, 2)

    n = len(A)

    v = np.random.rand(n, 1)
    v_moño = f(A, f(A, v))

    e = multMat(v_moño.T, v)

    k = 0
    while (np.abs(e - 1) > tol) and (k < K):
        v = v_moño
        v_moño = f(A, f(A, v))
        e = multMat(v_moño.T, v)
        k += 1

    autovalor = multMat(v_moño.T, multMat(A, v_moño))

    return (v_moño, autovalor[0][0], k)


def diagRH(A, tol=1e-15, K=1000):
    n = len(A)
    if n == 1:
        return np.eye(1), A

    v_1, lambda_1, _ = metpot2k(A, tol, K)

    I = np.eye(n, n)
    e_1 = I[:, [0]]
    u = e_1 - v_1
    norma_u = norma(u, 2) ** 2

    if norma_u > tol:
        proyeccion = multMat(u, u.T)
        H_v_1 = I - 2 * (proyeccion / norma_u)
    else:
        H_v_1 = I

    B = multMat(H_v_1, multMat(A, H_v_1.T))
    A_moño = B[1:, 1:]

    S_moño, D_moño = diagRH(A_moño, tol, K)

    D = np.zeros((n, n))
    D[0, 0] = lambda_1
    D[1:, 1:] = D_moño

    S = np.eye(n)
    S[1:, 1:] = S_moño
    S = multMat(H_v_1, S)

    return (S, D)


################# LABO 7 #################
def matriz(n):
    if n == 1:
        return np.array([[1]])

    T = np.random.rand(n, n)

    return T


def normalizarFilas(T):
    for i in range(0, len(T)):
        T[i] = T[i] / np.sum(T[i])

    return T


def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """

    T = matriz(n)

    T = normalizarFilas(T)

    return T.T


def transiciones_al_azar_uniformes(n, thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """

    T = matriz(n)

    for i in range(0, n):
        for j in range(0, n):
            if T[i, j] < thres:
                T[i, j] = 0
            else:
                T[i, j] = 1

    T = normalizarFilas(T)

    return T.T


def nucleo(A, tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """

    # Sabemos que A^t * A tiene el mismo nucleo que A. A^t * A es simetrica 
    # entonces podemos usar diagRH
    AtA = multMat(A.T, A)

    # Ahora la diagonalizamos
    S, D = diagRH(AtA, tol)

    nucleo = []

    for j in range(0, len(D)):
        if np.abs(D[j, j]) < tol:
            nucleo.append(S[:, j])

    return np.array(nucleo).T


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. 
    Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. 
      Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """

    rala = dict()

    if listado == []:
        return [rala, (m_filas, n_columnas)]

    for i in range(0, len(listado[0])):
        if not np.abs(listado[2][i]) < tol:
            rala[(listado[0][i], listado[1][i])] = listado[2][i]

    return [rala, (m_filas, n_columnas)]


def multiplica_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    w = np.zeros(shape=len(v))
    R = A[0]

    for (i, j) in R:
        w[i] += v[j] * R[(i, j)]

    return np.array([w])


################# LABO 8 #################

def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """

    m, n = np.shape(A)

    if m < n:
        U, Sigma2 = diagRH(multMat(A, A.T), tol)
        V = multMat(U.T, A).T
    else:
        V, Sigma2 = diagRH(multMat(A.T, A), tol)
        U = multMat(A, V)

    Sigma = []
    for i in range(min(n, m)):
        if np.abs(Sigma2[i, i]) > tol:
            Sigma.append(np.sqrt(Sigma2[i, i]))

    r = len(Sigma)

    if isinstance(k, int):
        r = min(r, k)
        Sigma = Sigma[:r]

    U = U[:, :r]
    V = V[:, :r]

    if m < n:
        for j in range(r):
            V[:, j] /= norma(V[:, j], 2)
    else:
        for j in range(r):
            U[:, j] /= norma(U[:, j], 2)

    return (U, Sigma, V)
