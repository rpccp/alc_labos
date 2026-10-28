import numpy as np


def copiar_matriz(A):
    """Crea una copia independiente de una matriz o un vector.

    Parámetros:
        A: matriz bidimensional o vector.
    Retorna:
        Una copia de A.
    """
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


def mult_mat(A, B):
    """Calcula el producto matricial entre dos matrices.

    Parámetros:
        A: primera matriz del producto.
        B: segunda matriz del producto.
    Retorna:
        La matriz resultante de multiplicar A por B.
    """
    filas_A, columnas_A = A.shape
    _, columnas_B = B.shape

    AB = np.zeros((filas_A, columnas_B))
    for i in range(filas_A):
        for j in range(columnas_B):
            for k in range(columnas_A):
                AB[i, j] += A[i, k] * B[k, j]

    return AB


def mult_mat_vector(A, x):
    """Calcula el producto de una matriz por un vector.

    Parámetros:
        A: matriz de coeficientes.
        x: vector de entrada.
    Retorna:
        El vector b tal que A x = b.
    """
    filas, columnas = A.shape
    b = np.zeros(filas)
    for i in range(filas):
        for j in range(columnas):
            b[i] += A[i, j] * x[j]

    return b


def producto_interno(x, y):
    """Calcula el producto interno entre dos vectores.

    Parámetros:
        x: primer vector.
        y: segundo vector.
    Retorna:
        El producto interno de x e y, o None si sus dimensiones no coinciden.
    """
    if len(x) != len(y):
        return None

    producto = 0
    for i in range(len(x)):
        producto += x[i] * y[i]

    return producto


def inversa(A):
    """Calcula la inversa de una matriz mediante su factorización LU.

    Parámetros:
        A: matriz cuadrada a invertir.
    Retorna:
        La matriz inversa de A, o None si A no se puede invertir.
    """
    L, U, _ = calculaLU(A)

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
        Una matriz con filas y columnas intercambiadas.
    """
    At = []
    filas, columnas = A.shape

    for i in range(columnas):
        nueva_fila = []

        for j in range(filas):
            nueva_fila.append(A[j, i])

        At.append(nueva_fila)

    return np.array(At)


def esSimetrica(A, tol=0):
    """Indica si A es simétrica.

    Parámetros:
        A: matriz cuadrada representada como un array bidimensional.
        tol: tolerancia absoluta para comparar entradas simétricas.
            Por defecto es cero, para comparar exactamente.
    Retorna:
        True si A coincide con su traspuesta dentro de tol; False en caso contrario.
    """
    filas, columnas = A.shape
    if filas != columnas:
        return False

    for i in range(filas):
        for j in range(columnas):
            if i != j:
                if abs(A[i, j] - A[j, i]) > tol:
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
    return abs(np.float32(x) - np.float32(y))


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
        True si tienen igual forma y sus elementos difieren a lo sumo en eps.
        False en caso contrario.
    """
    filas, columnas = A.shape
    if A.shape != B.shape:
        return False

    # TODO: NO PERMITIDO: np.finfo no figura entre las funciones habilitadas.
    eps = np.finfo('float').eps

    for i in range(filas):
        for j in range(columnas):
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
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = s[i]

    return A


def rota_y_escala(theta, s):
    """Combina el escalamiento indicado con una rotación.

    Parámetros:
        theta: ángulo de rotación en radianes.
        s: vector con los factores de escala.
    Retorna:
        La matriz que resulta de aplicar la rotación y el escalamiento.
    """
    return mult_mat(escala(s), rota(theta))


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
    w_hom = mult_mat_vector(T, v_hom)

    return w_hom[:2]


def norma(x, p):
    """Calcula la norma p de un vector.

    Parámetros:
        x: vector al que se le calcula la norma.
        p: orden de la norma, o 'inf' para la norma infinito.
    Retorna:
        El valor de la norma de x.
    """
    if p == 'inf':
        maximo = 0
        for valor in x.reshape(-1):
            if abs(valor) > maximo:
                maximo = abs(valor)
        return maximo
    suma = 0
    for valor in x.reshape(-1):
        suma += abs(valor) ** p
    return suma ** (1 / p)


def normaliza(X, p):
    """Normaliza los vectores de una lista según la norma p.

    Parámetros:
        X: lista de vectores a normalizar.
        p: orden de la norma, o 'inf' para la norma infinito.
    Retorna:
        Una lista con los vectores normalizados.
    """
    return [x / norma(x, p) for x in X]


def normaMatMC(A, q, p, Np):
    """Estima por Monte Carlo la norma matricial inducida de A.

    Parámetros:
        A: matriz a la que se le calcula la norma.
        q: orden de la norma del vector resultante.
        p: orden de la norma de los vectores de entrada.
        Np: cantidad de vectores aleatorios a probar.
    Retorna:
        La mayor norma q encontrada y el vector de norma p uno que la produjo.
        Es una estimación por muestreo; no garantiza el máximo global.
        Se supone Np positivo y al menos una columna en A.
    """
    res = 0
    m = A.shape[1]
    vector_maximo = np.zeros(m)
    hay_vector = False
    for _ in range(Np):
        v = np.zeros(m)
        for i in range(m):
            v[i] = np.random.uniform(-1.0, 1.0)
        normaV = norma(v, p)

        if normaV == 0:
            continue

        v /= normaV
        Av = mult_mat_vector(A, v)
        normaAv = norma(Av, q)

        # Conservamos también la primera muestra válida si A es la matriz nula.
        if normaAv > res or not hay_vector:
            res = normaAv
            vector_maximo = v
            hay_vector = True
    return res, vector_maximo


def normaExacta(A, p=[1, 'inf']):
    """Calcula la norma 1, la norma infinito o ambas para una matriz.

    Parámetros:
        A: matriz a la que se le calculan las normas.
        p: 1, 'inf' o la lista [1, 'inf'] para calcular ambas.
    Retorna:
        La norma solicitada, ambas normas en una lista, o None si p no es válido.
    """
    if p not in [1, 'inf'] and p != [1, 'inf']:
        return None

    filas, columnas = A.shape
    normaInf = 0
    norma1 = 0

    for row in range(filas):
        suma_fila = 0
        for col in range(columnas):
            suma_fila += abs(A[row, col])
        if suma_fila > normaInf: normaInf = suma_fila

    for col in range(columnas):
        suma_col = 0
        for row in range(filas):
            suma_col += abs(A[row, col])
        if suma_col > norma1: norma1 = suma_col

    if p == 1:
        return norma1
    elif p == 'inf':
        return normaInf
    return [norma1, normaInf]


def condMC(A, p):
    """Estima por Monte Carlo el número de condición de A.

    Parámetros:
        A: matriz cuadrada cuyo número de condición se estima.
        p: orden de la norma utilizada.
    Retorna:
        La estimación del número de condición, o None si A no es invertible.
    """
    Np = 1000
    A_inversa = inversa(A)
    if A_inversa is None:
        return None

    norma_A = normaMatMC(A, p, p, Np)[0]
    norma_A_inversa = normaMatMC(A_inversa, p, p, Np)[0]
    return norma_A * norma_A_inversa


def condExacto(A, p):
    """Calcula el número de condición de A usando la norma indicada.

    Parámetros:
        A: matriz cuadrada cuyo número de condición se calcula.
        p: 1, 'inf' o la lista [1, 'inf'] para calcular ambas condiciones.
    Retorna:
        El número de condición solicitado, una lista con ambas condiciones
        si p es [1, 'inf'], o None si p no es válido o A no es invertible.
    """
    if p not in [1, 'inf'] and p != [1, 'inf']:
        return None

    A_inversa = inversa(A)
    if A_inversa is None:
        return None

    norma_A = normaExacta(A, p)
    norma_A_inversa = normaExacta(A_inversa, p)
    if norma_A is None or norma_A_inversa is None:
        return None

    if p == [1, 'inf']:
        return [norma_A[0] * norma_A_inversa[0],
                norma_A[1] * norma_A_inversa[1]]

    return norma_A * norma_A_inversa


def calculaLU(A):
    """Calcula la factorización LU de una matriz sin pivoteo.

    Parámetros:
        A: matriz cuadrada a factorizar.
    Retorna:
        Las matrices L y U y la cantidad de operaciones realizadas. Retorna
        (None, None, 0) si A no es cuadrada o se necesita dividir por un pivote
        nulo. No realiza intercambios de filas.
    """
    if A is None:
        return None, None, 0

    n = A.shape[0]

    if A.shape[1] != n:
        return None, None, 0

    A_copia = copiar_matriz(A)
    L = np.eye(n)
    cant_op = 0

    for i in range(n - 1):
        pivot = A_copia[i, i]

        if pivot == 0:
            return None, None, 0

        for j in range(i + 1, n):
            coef = A_copia[j, i] / pivot
            cant_op += 1

            L[j, i] = coef
            A_copia[j, i] = 0

            for k in range(i + 1, n):
                A_copia[j, k] = A_copia[j, k] - coef * A_copia[i, k]
                cant_op += 2

    U = A_copia
    return L, U, cant_op


def res_tri(L, b, inferior=True):
    """Resuelve un sistema triangular mediante sustitución.

    Parámetros:
        L: matriz triangular.
        b: vector del lado derecho, compatible con L.
        inferior: indica si L es triangular inferior. Si no, se considera
            triangular superior.
    Retorna:
        El vector x que resuelve Lx = b, o None si la diagonal contiene
        un elemento nulo.
    """
    n = L.shape[0]
    X = np.zeros(len(b))

    if inferior:
        for i in range(n):
            if L[i, i] == 0:
                return None
            X[i] = b[i]
            for j in range(i):
                X[i] -= L[i, j] * X[j]
            X[i] /= L[i, i]
        return X

    for i in range(n - 1, -1, -1):
        if L[i, i] == 0:
            return None
        X[i] = b[i]
        for j in range(i + 1, n):
            X[i] -= L[i, j] * X[j]
        X[i] /= L[i, i]

    return X


def calculaLDV(A):
    """Calcula la factorización A = L D V a partir de la factorización LU.

    Parámetros:
        A: matriz cuadrada a factorizar.
    Retorna:
        Las matrices L, D y V, o None si no se puede calcular la factorización.
    """
    L, U, _ = calculaLU(A)

    if L is None or U is None:
        return None

    L2, D, _ = calculaLU(U.T)
    if L2 is None or D is None:
        return None

    return L, D, L2.T


def esSDP(A, atol=1e-8):
    """Determina si una matriz es simétrica definida positiva.

    Parámetros:
        A: matriz cuadrada.
        atol: tolerancia máxima para considerar simétricas dos entradas.
    Retorna:
        True si A es simétrica dentro de la tolerancia y definida positiva.
        False en caso contrario.
    """
    if A is None or atol < 0 or len(A.shape) != 2:
        return False

    n = A.shape[0]
    if A.shape[1] != n:
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j] - A[j, i]) > atol:
                return False

    resultado_ldv = calculaLDV(A)
    if resultado_ldv is None:
        return False
    _, D, _ = resultado_ldv

    for i in range(n):
        if D[i, i] <= atol:
            return False

    return True


def calculaCholesky(A, atol=1e-10):
    """Calcula una factorización de Cholesky de una matriz SDP.

    Parámetros:
        A: matriz simétrica definida positiva.
        atol: tolerancia usada para verificar simetría y positividad.
    Retorna:
        La matriz triangular inferior R tal que A = R R.T, o None si A no es
        simétrica definida positiva.
    """
    if not esSDP(A, atol):
        return None

    resultado_ldv = calculaLDV(A)
    if resultado_ldv is None:
        return None
    L, D, _ = resultado_ldv

    n = L.shape[0]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            R[i, j] = L[i, j] * D[j, j] ** 0.5

    return R


def GS(A, tol, retorna_nops):
    """Calcula una factorización QR mediante Gram-Schmidt modificado.

    Parámetros:
        A: matriz de tamaño (m, n).
        tol: tolerancia para descartar residuos de norma pequeña.
        retorna_nops: indica si se devuelve la estimación de operaciones.
    Retorna:
        Q de tamaño (m, r) y R de tamaño (r, n), donde r es la cantidad
        de columnas independientes detectadas. Opcionalmente devuelve
        nops como tercer resultado. (Cuenta productos, sumas, restas,
        divisiones y raíces)
    """
    filas, columnas = A.shape
    Q = np.zeros((filas, columnas))
    R = np.zeros((columnas, columnas))
    r = 0
    nops = 0

    # Procesamos cada columna...
    for j in range(columnas):
        q_j_moño = copiar_matriz(A[:, j])

        # Proyectamos solo sobre los r vectores de la base construida.
        for k in range(r):
            R[k, j] = producto_interno(Q[:, k], q_j_moño)
            q_j_moño = q_j_moño - R[k, j] * Q[:, k]
            nops += (2 * filas - 1) + 2 * filas

        norma_qj_moño = norma(q_j_moño, 2)
        nops += 2 * filas

        # Solo un residuo no nulo aporta un nuevo vector a la base.
        if norma_qj_moño > tol:
            R[r, j] = norma_qj_moño
            Q[:, r] = q_j_moño / R[r, j]
            nops += filas
            r += 1

    Q_final = Q[:, :r]
    R_final = R[:r, :]

    if retorna_nops:
        return Q_final, R_final, nops

    return Q_final, R_final


def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna Q de tamaño (n, r) y R de tamaño (r, n), calculadas con Gram Schmidt,
    siendo r la cantidad de columnas independientes detectadas. Como tercer
    argumento opcional retorna la estimación del número de operaciones.
    Si la matriz A no es de n x n, debe retornar None
    """
    if A is None: return None
    if len(A) != len(A[0]): return None

    return GS(A, tol, retorna_nops)


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    A una matriz de m x n.
    tol la tolerancia para considerar nulo el vector que define Householder.
    retorna matrices Q y R calculadas con reflexiones de Householder
    Admite tanto m>=n como m<n.
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve: 
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    Q tiene tamaño (m, m) y R tiene tamaño (m, n), con A = QR.
    La diagonal de R se devuelve no negativa.
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso
    """
    filas, columnas = A.shape

    R = copiar_matriz(A)

    Q = np.eye(filas)

    extra_info = {
        'R_matrices': [],
        'Q_matrices': []
    }

    for k in range(min(filas, columnas)):
        if extras:
            extra_info['R_matrices'].append(copiar_matriz(R))
            extra_info['Q_matrices'].append(copiar_matriz(Q))

        # tajada vertical de la submatriz abajo a la derecha a partir de R[k,k]
        x = R[k:, k]

        signo_x0 = 1 if x[0] >= 0 else -1
        alpha = -signo_x0 * norma(x, 2)

        # definimos u perpendicular al espejo
        # columna x -  normaSignoAlternado* primer canonico con dimension (m-k+1)
        u = x - alpha * np.eye(filas - k)[0]
        norma_u = norma(u, 2)
        if norma_u > tol:  # si no lo es, no lo procesamos, ya habia ceros debajo de la diagonal, pasamos a sig columna
            # si norma mayor a cero, reflejamos
            # normalizo
            u = u / norma_u
            # submatriz de reflectores espejo, dimension (m-k)x(m-k)
            u_columna = u.reshape((filas - k, 1))
            H_k = np.eye(filas - k) - 2 * mult_mat(u_columna, u_columna.T)
            # expandimos a la dimension de la matriz original, agrgando identidad arriba a la izquierda
            H_kp = np.eye(filas)
            H_kp[k:filas, k:filas] = H_k
            R = mult_mat(H_kp, R)
            Q = mult_mat(Q, H_kp)

    # Cambiar ambos signos conserva Q R y hace no negativa la diagonal de R.
    for k in range(min(filas, columnas)):
        if R[k, k] < 0:
            R[k, :] = -R[k, :]
            Q[:, k] = -Q[:, k]

    if extras:
        extra_info['R_matrices'].append(copiar_matriz(R))
        extra_info['Q_matrices'].append(copiar_matriz(Q))
        return Q, R, extra_info

    return Q, R


def calculaQR(A, metodo="RH", tol=1e-12):
    if metodo == "GS":
        return GS(A, tol, retorna_nops=False)
    elif metodo == "RH":
        return QR_con_HH(A, tol)
    else:
        return None


def metpot2k(A, tol=1e-15, K=1000):
    """Aplica dos pasos de potencia por iteración, hasta tol o K iteraciones.

    Parámetros:
        A: matriz real cuadrada.
        tol: tolerancia sobre |v_nuevo.T v_anterior - 1|.
        K: máximo de iteraciones dobles.
    Retorna:
        Vector unitario v de forma (n,), cociente de Rayleigh lambda y número
        de iteraciones. Si Av=0 devuelve v y lambda=0 sin dividir por cero.
    """
    A = np.array(A, dtype=float)
    v = np.random.rand(A.shape[0], 1)
    v = v / norma(v, 2)
    iteraciones = 0

    for paso in range(int(K)):
        # Primer paso: una imagen nula identifica un autovector de autovalor cero.
        Av = mult_mat(A, v)
        norma_Av = norma(Av, 2)
        if norma_Av == 0:
            return v[:, 0], 0.0, iteraciones
        intermedio = Av / norma_Av

        # Segundo paso: también se contempla A^2 v=0 con Av distinto de cero.
        siguiente = mult_mat(A, intermedio)
        norma_siguiente = norma(siguiente, 2)
        if norma_siguiente == 0:
            return intermedio[:, 0], 0.0, paso + 1
        siguiente = siguiente / norma_siguiente
        error = abs(producto_interno(siguiente[:, 0], v[:, 0]) - 1)
        v = siguiente
        iteraciones = paso + 1
        if error <= tol:
            break

    Av = mult_mat(A, v)
    autovalor = producto_interno(v[:, 0], Av[:, 0])
    norma_Av = norma(Av, 2)
    residuo = norma(Av - autovalor * v, 2)
    # A^2 puede estabilizar una mezcla de autovectores de +lambda y -lambda.
    # Solo si se estabilizó la dirección pero no satisface Av=lambda*v,
    # proyectamos esa mezcla sobre uno de los dos autoespacios.
    if iteraciones and error <= tol and norma_Av > 0:
        if residuo > 10 * tol ** 0.5 * norma_Av:
            candidato = v + Av / norma_Av
            norma_candidato = norma(candidato, 2)
            if norma_candidato > 0:
                v = candidato / norma_candidato
                Av = mult_mat(A, v)
                autovalor = producto_interno(v[:, 0], Av[:, 0])
    return v[:, 0], autovalor, iteraciones


def diagRH(A, tol=1e-15, K=1000):
    """Diagonaliza A simétrica mediante potencia y reflexiones de Householder.

    Parámetros:
        A: matriz real cuadrada y simétrica.
        tol: tolerancia de potencia y comprobación numérica de simetría.
        K: máximo de iteraciones dobles de potencia por autovector.
    Retorna:
        S y D, con A aproximadamente S D S.T; None si A no es simétrica.
    """
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    n = len(A)
    if n <= 1:
        return np.eye(n), np.array(A)
    escala = norma(A, 'inf')
    if not esSimetrica(A, tol * max(1.0, escala)):
        return None
    if escala == 0:
        return np.eye(n), np.zeros((n, n))

    v_1, lambda_1, _ = metpot2k(A, tol, K)
    v_1 = v_1.reshape((n, 1))
    # El signo del autovector es libre; esta elección evita e1-v casi nulo.
    if v_1[0, 0] > 0:
        v_1 = -v_1

    I = np.eye(n, n)
    e_1 = I[:, [0]]
    u = e_1 - v_1
    norma_u = norma(u, 2) ** 2

    if norma_u > tol:
        proyeccion = mult_mat(u, u.T)
        H_v_1 = I - 2 * (proyeccion / norma_u)
    else:
        H_v_1 = I

    B = mult_mat(H_v_1, mult_mat(A, H_v_1.T))
    # Recuperamos la simetría que puede perderse por redondeo en los productos.
    B = (B + B.T) / 2
    A_moño = B[1:, 1:]

    S_moño, D_moño = diagRH(A_moño, tol, K)

    D = np.zeros((n, n))
    D[0, 0] = lambda_1
    D[1:, 1:] = D_moño

    S = np.eye(n)
    S[1:, 1:] = S_moño
    S = mult_mat(H_v_1, S)

    return S, D


def matriz(n):
    if n == 1:
        return np.array([[1]])

    T = np.random.rand(n, n)

    return T


def normalizarFilas(T):
    """Normaliza las filas de una matriz de pesos de transición.

    Parámetros:
        T: matriz cuadrada de pesos no negativos; se modifica en el lugar.
    Retorna:
        T con filas que suman uno. Si una fila no tiene salidas, agrega
        un lazo al mismo estado antes de normalizar. Así, al trasponer,
        no quedan columnas nulas.
    """
    for i in range(0, len(T)):
        suma = 0
        for j in range(T.shape[1]):
            suma += T[i, j]
        if suma == 0:
            T[i, i] = 1
            suma = 1
        T[i] = T[i] / suma

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
    Si un estado queda sin salidas, se agrega una transición a sí mismo.
    """

    T = matriz(n)

    for i in range(0, n):
        for j in range(0, n):
            if T[i, j] <= thres:
                T[i, j] = 1
            else:
                T[i, j] = 0

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
    AtA = mult_mat(A.T, A)

    # Ahora la diagonalizamos
    S, D = diagRH(AtA, tol)

    nucleo = []

    for j in range(0, len(D)):
        if abs(D[j, j]) <= tol:
            nucleo.append(S[:, j])

    return np.array(nucleo).T


def puntos_fijos(T, tol):
    """Calcula una base de puntos fijos de una matriz de transición.

    Parámetros:
        T: matriz de transición de tamaño (n, n); no se modifica.
        tol: tolerancia utilizada por nucleo para detectar autovalores nulos.
    Retorna:
        Matriz de tamaño (n, k) cuyas columnas son puntos fijos independientes,
        normalizados con norma 1 igual a uno. Se orientan con suma no negativa;
        una base del núcleo puede contener componentes de signos distintos.
    """
    n = T.shape[0]
    P = nucleo(T - np.eye(n), tol)
    if P.size == 0:
        return np.zeros((n, 0))

    for j in range(P.shape[1]):
        P[:, j] = P[:, j] / norma(P[:, j], 1)
        suma = 0
        for i in range(n):
            suma += P[i, j]
        if suma < 0:
            P[:, j] = -P[:, j]

    return P


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
        if not abs(listado[2][i]) < tol:
            rala[(listado[0][i], listado[1][i])] = listado[2][i]

    return [rala, (m_filas, n_columnas)]


def multiplica_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w de longitud m_filas, resultado de multiplicar A con v.
    """
    w = np.zeros(A[1][0])
    R = A[0]

    for (i, j) in R:
        w[i] += v[j] * R[(i, j)]

    return w


def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """

    m, n = np.shape(A)

    if m < n:
        U, Sigma2 = diagRH(mult_mat(A, A.T), tol)
        V = mult_mat(U.T, A).T
    else:
        V, Sigma2 = diagRH(mult_mat(A.T, A), tol)
        U = mult_mat(A, V)

    Sigma = []
    for i in range(min(n, m)):
        if abs(Sigma2[i, i]) > tol:
            Sigma.append(Sigma2[i, i] ** 0.5)

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

    return U, Sigma, V
