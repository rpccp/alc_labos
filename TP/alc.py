"""Funciones del trabajo práctico Calibración y Pose de la Cámara."""

import numpy as np


def obtenerQ2d(imgPts, objPts):
    """Construye la matriz Q del DLT coplanar (ecuación 11 del enunciado).

    Parámetros:
        imgPts : arreglo de tamaño (n, 2)
            Coordenadas (u, v) de los puntos en la imagen, en píxeles.
        objPts : arreglo de tamaño (n, 2)
            Coordenadas (U, V) de los puntos en el plano del objeto, expresadas
            en una misma unidad. La fila i corresponde al mismo punto en ambos
            arreglos. Se reciben coordenadas cartesianas.

    Retorna:
        Q : numpy.ndarray de tamaño (2*n, 9)
            Matriz del sistema Qh = 0, con los coeficientes de H ordenados por
            filas en h. Cada correspondencia aporta primero la ecuación de u
            y después la de v.
    """
    # Convertimos a arreglos de punto flotante.
    puntos_imagen = np.asarray(imgPts, dtype=float)
    puntos_objeto = np.asarray(objPts, dtype=float)

    cantidad_puntos = puntos_imagen.shape[0]
    Q = np.zeros((2 * cantidad_puntos, 9))

    # Apilamos las dos ecuaciones de cada correspondencia según la ecuación 11.
    for indice in range(cantidad_puntos):
        u, v = puntos_imagen[indice]
        U, V = puntos_objeto[indice]

        Q[2 * indice] = [U, V, 1, 0, 0, 0, -u * U, -u * V, -u]
        Q[2 * indice + 1] = [0, 0, 0, U, V, 1, -v * U, -v * V, -v]

    return Q


def norma(x: np.ndarray, p: int | str) -> float:
    """Recibe un vector x y un número p, admite p numérico o "inf".

    Devuelve su norma p.

    Reutilizada del módulo ALC."""
    if p == 1:
        return sum([abs(n) for n in x])

    elif p == "inf":
        max = 0
        for n in x:
            if abs(n) > max:
                max = abs(n)
        return max

    suma = sum([n**p for n in x])
    return suma ** (1 / p)


def mult_mat(A, B):
    """Recibe A (m, n) y B (n, p).

    Devuelve su producto (m, p).

    Reutilizada del módulo ALC."""
    filasA = A.shape[0]
    columnasB = B.shape[1]
    columnasA = A.shape[1]
    AB = np.zeros((filasA, columnasB))

    for i in range(filasA):
        for j in range(columnasB):
            for k in range(columnasA):
                AB[i, j] += A[i, k] * B[k, j]

    return AB

def calculaLU(A):
    """Calcula la factorización LU de una matriz cuadrada.

    Parámetros
    ----------
    A : arreglo de tamaño (n, n)
        Matriz cuadrada a factorizar.

    Retorna
    -------
    matriz_L : numpy.ndarray de tamaño (n, n)
        Matriz triangular inferior con unos en la diagonal principal.
        Retorna None si la matriz no cumple los requisitos o hay ceros en la diagonal.
    matriz_U : numpy.ndarray de tamaño (n, n)
        Matriz triangular superior escalonada.
        Retorna None si la matriz no cumple los requisitos o hay ceros en la diagonal.
    cant_operaciones : int
        Cantidad de operaciones elementales realizadas. Retorna 0 si el proceso falla.
    """
    if A is None:
        return None, None, 0

    # Intentamos convertir la entrada a un array de floats; si los tipos son incompatibles, abortamos
    try:
        matriz_U = np.array(A, dtype=float)
    except:
        return None, None, 0

    # Verificamos que la matriz sea estrictamente bidimensional y cuadrada
    if matriz_U.ndim != 2 or matriz_U.shape[0] != matriz_U.shape[1]:
        return None, None, 0

    dimension = matriz_U.shape[0]
    matriz_L = np.eye(dimension)
    cant_operaciones = 0

    # Aplicamos eliminación gaussiana para construir L (multiplicadores) y escalonar U
    for fila_pivote in range(dimension - 1):
        pivote = matriz_U[fila_pivote][fila_pivote]

        # Abortamos si encontramos un cero en la diagonal, ya que no implementamos pivoteo
        if pivote == 0:
            return None, None, 0

        for fila_actual in range(fila_pivote + 1, dimension):
            multiplicador = matriz_U[fila_actual][fila_pivote] / pivote
            cant_operaciones += 1

            matriz_L[fila_actual][fila_pivote] = multiplicador
            matriz_U[fila_actual][fila_pivote] = 0

            matriz_U[fila_actual][fila_pivote + 1:] = matriz_U[fila_actual][fila_pivote + 1:] - multiplicador * matriz_U[fila_pivote][fila_pivote + 1:]

            cant_operaciones += 2 * (dimension - fila_pivote - 1)

    return matriz_L, matriz_U, cant_operaciones


def res_tri(A, b, inferior=True):
    """Resuelve el sistema Ax = b cuando A es triangular inferior o superior.

    Parámetros
    ----------
    A : arreglo de tamaño (n, n)
        Matriz del sistema, debe ser triangular.
    b : arreglo de tamaño (n,)
        Vector de términos independientes.
    inferior : bool, opcional
        Indica si la matriz es triangular inferior (True) o superior (False).
        Por defecto es True.

    Retorna
    -------
    vector_solucion : numpy.ndarray de tamaño (n,)
        Vector con la solución del sistema. Retorna None si las dimensiones
        no son compatibles o si la matriz es singular.
    """
    # Intentamos convertir las entradas a arrays numéricos; abortamos si los tipos son incompatibles
    try:
        matriz_T = np.array(A, dtype=float)
        vector_b = np.array(b, dtype=float)
    except:
        return None

    # Verificamos que la matriz sea bidimensional y estrictamente cuadrada
    if matriz_T.ndim != 2 or matriz_T.shape[0] != matriz_T.shape[1]:
        return None

    # Verificamos que el término independiente sea un vector unidimensional
    if vector_b.ndim != 1:
        return None

    dimension = matriz_T.shape[0]

    # Chequeamos la compatibilidad de dimensiones entre la matriz y el vector
    if vector_b.shape[0] != dimension:
        return None

    vector_solucion = np.zeros(dimension)

    if inferior:
        # Resolvemos por sustitución hacia adelante (forward substitution)
        for fila in range(dimension):
            # Si encontramos un cero en la diagonal, la matriz es singular (no tiene solución única)
            if matriz_T[fila][fila] == 0:
                return None

            suma_parcial = 0.0
            for columna in range(fila):
                suma_parcial += matriz_T[fila][columna] * vector_solucion[columna]

            vector_solucion[fila] = (vector_b[fila] - suma_parcial) / matriz_T[fila][fila]
    else:
        # Resolvemos por sustitución hacia atrás (backward substitution)
        for fila in range(dimension - 1, -1, -1):
            # Si encontramos un cero en la diagonal, la matriz es singular (no tiene solución única)
            if matriz_T[fila][fila] == 0:
                return None

            suma_parcial = 0.0
            for columna in range(fila + 1, dimension):
                suma_parcial += matriz_T[fila][columna] * vector_solucion[columna]

            vector_solucion[fila] = (vector_b[fila] - suma_parcial) / matriz_T[fila][fila]

    return vector_solucion


def inversa(A):
    """Calcula la inversa de A mediante su factorización LU.

    Parámetros
    ----------
    A : arreglo de tamaño (n, n)
        Matriz cuadrada a invertir.

    Retorna
    -------
    matriz_inversa : numpy.ndarray de tamaño (n, n)
        Matriz inversa calculada resolviendo sistemas triangulares.
        Retorna None si la factorización falla o la matriz es singular.
    """
    # Obtenemos la descomposición LU de la matriz original
    resultado_lu = calculaLU(A)
    matriz_L, matriz_U, _ = resultado_lu

    # Verificamos que la matriz haya podido factorizarse correctamente
    if matriz_L is None or matriz_U is None:
        return None

    dimension = matriz_U.shape[0]
    matriz_inversa = np.zeros((dimension, dimension))

    # Construimos la inversa resolviendo el sistema para cada columna de la matriz identidad
    for columna in range(dimension):
        # Generamos el vector de la base canónica (e_j) correspondiente a la columna actual
        vector_canonico = np.zeros(dimension)
        vector_canonico[columna] = 1.0

        # Resolvemos el sistema triangular inferior L * y = e_j
        vector_y = res_tri(matriz_L, vector_canonico)
        if vector_y is None:
            return None

        # Resolvemos el sistema triangular superior U * x = y
        columna_inversa = res_tri(matriz_U, vector_y, inferior=False)
        if columna_inversa is None:
            return None

        # Acoplamos el vector resultante como columna en la matriz inversa final
        matriz_inversa[:, columna] = columna_inversa

    return matriz_inversa


def metpot2k(A, tol = 1e-15, K = 1000):
    """Recibe A simétrica, tolerancia tol y máximo K de iteraciones.

    Devuelve un autovector dominante, su autovalor y las iteraciones usadas.

    Reutilizada del módulo ALC."""

    def f(A, v):
        res = mult_mat(A, v)
        return res / norma(res, 2)

    n = len(A)

    v = np.random.rand(n, 1)
    v_moño = f(A, f(A, v))

    e = mult_mat(v_moño.T, v)

    k = 0
    while (np.abs(e - 1) > tol) and (k < K):
        v = v_moño
        v_moño = f(A, f(A, v))
        e = mult_mat(v_moño.T, v)
        k += 1

    autovalor = mult_mat(v_moño.T, mult_mat(A, v_moño))

    return (v_moño, autovalor[0][0], k)


def diagRH(A, tol = 1e-15, K = 1000):
    """Recibe A simétrica, tolerancia tol y máximo K de iteraciones.

    Devuelve S y D tales que A es aproximadamente S D S.T: las columnas
    de S son autovectores y la diagonal de D contiene sus autovalores.

    Reutilizada del módulo ALC."""

    n = len(A)
    if n == 1:
        return np.eye(1), A

    v_1, lambda_1, _ = metpot2k(A, tol, K)

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
    A_moño = B[1:, 1:]

    S_moño, D_moño = diagRH(A_moño, tol, K)

    D = np.zeros((n, n))
    D[0, 0] = lambda_1
    D[1:, 1:] = D_moño

    S = np.eye(n)
    S[1:, 1:] = S_moño
    S = mult_mat(H_v_1, S)

    return (S, D)


def resolverDLT2D(Q, tol=1e-15, K=1000):
    """Obtiene una homografía H a partir de la matriz Q del DLT coplanar.

    Parámetros
    ----------
    Q : matriz de tamaño (2*n, 9)
        Para datos reales, es mejor construir Q con coordenadas normalizadas.
    tol : float
        Tolerancia utilizada por diagRH.
    K : int
        Máximo de iteraciones del método de potencia en cada paso de diagRH.

    Retorna
    -------
    H : numpy.ndarray de tamaño (3, 3)
        Homografía del plano del objeto a la imagen, salvo escala.
        Si Q usa coordenadas normalizadas, H relaciona esas coordenadas
        tenemos que desnormalizarla después.

    Resuelve min ||Qh||_2 con ||h||_2=1: h es un autovector asociado
    al menor autovalor de Q.T Q. No normaliza puntos ni desnormaliza H.
    """
    # Construimos la matriz simétrica.
    Q = np.asarray(Q, dtype=float)
    B = mult_mat(Q.T, Q)
    S, D = diagRH(B, tol, K)

    # Buscamos el menor autovalor.
    indice_minimo = 0
    for indice in range(1, D.shape[0]):
        if D[indice, indice] < D[indice_minimo, indice_minimo]:
            indice_minimo = indice

    # La columna que nos queda contiene los 9 coeficientes de H.
    h = S[:, indice_minimo]
    h = h / norma(h, 2)
    return h.reshape((3, 3))


def normalizarPuntos2d(puntos):
    """Aplica la normalización isotrópica a puntos de un plano.

    Parámetros
    ----------
    puntos : arreglo de tamaño (n, 2)
        Coordenadas cartesianas. Espera al menos dos puntos
        distintos, para que la distancia media al centroide sea positiva.

    Retorna
    -------
    puntos_normalizados : numpy.ndarray de tamaño (n, 2)
        Puntos con centroide en el origen y distancia media raíz de 2 al origen.
    T : numpy.ndarray de tamaño (3, 3)
        Transformación que lleva puntos originales a normalizados en
        coordenadas homogéneas: p_normalizado = T p_original.
    """
    # Centramos el conjunto restando su centroide a cada punto.
    puntos = np.asarray(puntos, dtype=float)
    centroide = np.mean(puntos, axis=0)
    puntos_centrados = puntos - centroide

    # La misma escala en ambos ejes fija la distancia media en raíz de 2.
    distancias = np.sqrt(np.sum(puntos_centrados ** 2, axis=1))
    escala = np.sqrt(2) / np.mean(distancias)
    puntos_normalizados = escala * puntos_centrados
    T = np.array([[escala, 0, -escala * centroide[0]],
                  [0, escala, -escala * centroide[1]],
                  [0, 0, 1]])
    return puntos_normalizados, T


def estimarHomografia(imgPts, objPts, tol=1e-15, K=1000):
    """Estima la homografía del plano del objeto a la imagen con DLT normalizado.

    Parámetros
    ----------
    imgPts : arreglo de tamaño (n, 2)
        Coordenadas (u, v) de los puntos en la imagen, en píxeles.
    objPts : arreglo de tamaño (n, 2)
        Coordenadas (U, V) de los puntos del plano. Las filas deben
        corresponder a los mismos puntos que las filas de imgPts.
    tol : float
        Tolerancia de la diagonalización usada por resolverDLT2D.
    K : int
        Máximo de iteraciones del método de la potencia en cada paso de diagRH.

    Retorna
    -------
    H : numpy.ndarray de tamaño (3, 3)
        Homografía en las coordenadas originales: p_imagen ~ H p_objeto.
        Se devuelve salvo un factor de escala.
    """
    # Normalizamos cada conjunto por separado antes de construir el sistema.
    puntos_imagen, T_img = normalizarPuntos2d(imgPts)
    puntos_objeto, T_obj = normalizarPuntos2d(objPts)
    Q = obtenerQ2d(puntos_imagen, puntos_objeto)
    H_normalizada = resolverDLT2D(Q, tol, K)

    # Invertimos la escala y traslación de T_img.
    escala = T_img[0, 0]
    T_img_inversa = np.array([
        [1 / escala, 0, -T_img[0, 2] / escala],
        [0, 1 / escala, -T_img[1, 2] / escala],
        [0, 0, 1]])

    # Volvemos a las coordenadas originales según la ecuación 14.
    H = mult_mat(T_img_inversa, mult_mat(H_normalizada, T_obj))
    return H


def armarA(Hlst):
    """Construye la matriz A del sistema homogéneo de calibración coplanar.

    Parámetros:
        Hlst : lista de N matrices de tamaño (3, 3)
            Homografías del plano del objeto a la imagen.

    Retorna:
        A : numpy.ndarray de tamaño (2*N, 6)
            Matriz del sistema Ab=0, con las incógnitas ordenadas como
            b=(B11, B12, B22, B13, B23, B33).
    """
    def restriccion(H, i, j):
        # Coeficientes de c_i.T B c_j, agrupando las entradas simétricas de B.
        return np.array([
            H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[1, i] * H[1, j],
            H[2, i] * H[0, j] + H[0, i] * H[2, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j]])

    A = np.zeros((2 * len(Hlst), 6))
    for i in range(len(Hlst)):
        H = Hlst[i]
        A[2 * i] = restriccion(H, 0, 1)
        A[2 * i + 1] = restriccion(H, 0, 0) - restriccion(H, 1, 1)
    return A
