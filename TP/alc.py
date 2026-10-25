"""Funciones del trabajo práctico Calibración y Pose de la Cámara."""

import numpy as np


def obtenerQ2d(imgPts, objPts):
    """Construye la matriz Q del DLT coplanar (ecuación 11 del enunciado).

    Parámetros
    ----------
    imgPts : arreglo de tamaño (n, 2)
        Coordenadas (u, v) de los puntos en la imagen, en píxeles.
    objPts : arreglo de tamaño (n, 2)
        Coordenadas (U, V) de los puntos en el plano del objeto, expresadas
        en una misma unidad. La fila i corresponde al mismo punto en ambos
        arreglos. Se reciben coordenadas cartesianas.

    Retorna
    -------
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


def multMat(A, B):
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


def metpot2k(A, tol = 1e-15, K = 1000):
    """Recibe A simétrica, tolerancia tol y máximo K de iteraciones.

    Devuelve un autovector dominante, su autovalor y las iteraciones usadas.

    Reutilizada del módulo ALC."""

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
    B = multMat(Q.T, Q)
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
    H = multMat(T_img_inversa, multMat(H_normalizada, T_obj))
    return H
