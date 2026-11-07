"""Funciones del trabajo práctico Calibración y Pose de la Cámara."""

import numpy as np

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

def obtenerQ3d(imgPts, objPts):
    """Construye la matriz Q del DLT 3D (ecuación 6 del enunciado).

    Parámetros:
        imgPts : arreglo de tamaño (n, 2)
            Coordenadas (u, v) de los puntos en la imagen, en píxeles.
        objPts : arreglo de tamaño (n, 3)
            Coordenadas cartesianas (U, V, W) de los puntos en el mundo real.
            La fila i corresponde al mismo punto en ambos arreglos.

    Retorna:
        Q : numpy.ndarray de tamaño (2*n, 12)
            Matriz del sistema Qm = 0, con los coeficientes de M ordenados por
            filas en m. Cada correspondencia aporta primero la ecuación de u
            y después la de v.
    """
    # Verificamos que las dimensiones y cantidades coincidan
    cantidad_puntos = np.shape(imgPts)[0]
    
    if np.shape(objPts)[0] != cantidad_puntos:
        return None
    if np.shape(imgPts)[1] != 2 or np.shape(objPts)[1] != 3:
        return None

    # Convertimos a arreglos de punto flotante para evitar errores de tipo
    puntos_imagen = np.asarray(imgPts, dtype=float)
    puntos_objeto = np.asarray(objPts, dtype=float)

    # Inicializamos Q con ceros
    Q = np.zeros((2 * cantidad_puntos, 12))

    # Apilamos las dos ecuaciones de cada correspondencia según la ecuación 6
    for indice in range(cantidad_puntos):
        u, v = puntos_imagen[indice]
        U, V, W = puntos_objeto[indice]

        Q[2 * indice]     = [U, V, W, 1, 0, 0, 0, 0, -u * U, -u * V, -u * W, -u]
        Q[2 * indice + 1] = [0, 0, 0, 0, U, V, W, 1, -v * U, -v * V, -v * W, -v]

    return Q

def resolverDLT3D(Q, tol=1e-15, K=1000):
    """Resuelve el sistema Qm = 0 para encontrar la matriz de proyección M.

    Parámetros:
        Q : numpy.ndarray de tamaño (2*n, 12)
            Matriz de diseño del sistema obtenida a partir de las correspondencias 2D-3D.
        tol : float
            Tolerancia utilizada por diagRH.
        K : int
            Máximo de iteraciones del método de potencia en cada paso de diagRH.

    Retorna:
        M : numpy.ndarray de tamaño (3, 4)
            Matriz de proyección de la cámara calculada mediante cuadrados mínimos.
    """
    # Construimos la matriz simétrica (Q^T * Q).
    Q = np.asarray(Q, dtype=float)
    B = mult_mat(Q.T, Q)
    
    # Calculamos autovectores (S) y autovalores (D) de la matriz B.
    S, D = diagRH(B, tol, K)
    
    # Buscamos explícitamente el índice del menor autovalor en la diagonal.
    indice_minimo = 0
    for indice in range(1, D.shape[0]):
        if D[indice, indice] < D[indice_minimo, indice_minimo]:
            indice_minimo = indice
            
    # El vector que minimiza el error es la columna de S asociada al menor autovalor.
    m = S[:, indice_minimo]
    
    # Forzamos que la norma sea exactamente 1 para cumplir la restricción del DLT.
    m = m / norma(m, 2)
    
    # Reorganizamos el vector de 12 elementos en la matriz de la cámara de 3x4.
    M = m.reshape((3, 4))
    
    return M


def devolverMatricesCalibracion3d(M, fuerza_k12_0=False, fuerza_au_av_iguales=False):
    """Descompone la matriz de proyección M en los parámetros intrínsecos (K) y extrínsecos (R, t).

    Parámetros:
        M : numpy.ndarray de tamaño (3, 4)
            Matriz de proyección de la cámara calculada previamente.
        fuerza_k12_0 : bool, opcional
            Si es True, impone que gamma (K12) sea igual a 0.
        fuerza_au_av_iguales : bool, opcional
            Si es True, impone que au sea igual a av.

    Retorna:
        R : numpy.ndarray de tamaño (3, 3)
            Matriz ortogonal de rotación de la cámara.
        t : numpy.ndarray de tamaño (3,)
            Vector de traslación espacial.
        K : numpy.ndarray de tamaño (3, 3)
            Matriz de parámetros intrínsecos de la cámara (triangular superior).
    """
    # Verificamos que la matriz tenga exactamente 3 filas y 4 columnas
    if M.shape != (3, 4):
        return None, None, None

    # Extraemos los primeros tres elementos de cada fila (qi) y el último elemento (mi4)
    q1 = M[0, :3]
    q2 = M[1, :3]
    q3 = M[2, :3]

    m14 = M[0, 3]
    m24 = M[1, 3]
    m34 = M[2, 3]

    # Calculamos el factor de escala p (inicialmente positivo)
    p = 1.0 / norma(q3, 2)

    # Despejamos la tercera fila de la matriz de rotación
    r3 = p * q3

    # Calculamos v0 aprovechando la ortogonalidad con r3
    v0 = p * producto_interno(q2, r3)

    # Aislamos la segunda fila de rotación, calculamos av y normalizamos
    r2_sin_norma = (p * q2) - (v0 * r3)
    av = norma(r2_sin_norma, 2)
    r2 = r2_sin_norma / av

    # Calculamos u0 y el sesgo (gamma) usando la ortogonalidad
    u0 = p * producto_interno(q1, r3)
    gamma = p * producto_interno(q1, r2)

    # Aislamos la primera fila de rotación, calculamos au y normalizamos
    r1_sin_normalizar = (p * q1) - (gamma * r2) - (u0 * r3)
    au = norma(r1_sin_normalizar, 2)
    r1 = r1_sin_normalizar / au

    # Armamos la matriz ortogonal de rotación R
    R = np.array([r1, r2, r3])

    # Verificamos que R sea una rotación propia (det(R) = +1) y no una reflexión
    det_R = (R[0, 0] * (R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1])
           - R[0, 1] * (R[1, 0] * R[2, 2] - R[1, 2] * R[2, 0])
           + R[0, 2] * (R[1, 0] * R[2, 1] - R[1, 1] * R[2, 0]))

    # Si es reflexión, invertimos el signo del factor de escala y de la matriz R
    if det_R < 0:
        p = -p
        R = -R

    # Imponemos las restricciones opcionales del inciso 2d sobre los parámetros de K
    if fuerza_k12_0:
        gamma = 0.0

    if fuerza_au_av_iguales:
        prom = 0.5 * (au + av)
        au = prom
        av = prom

    # Armamos la matriz intrínseca K
    K = np.array([
        [au, gamma, u0],
        [0.0, av, v0],
        [0.0, 0.0, 1.0]
    ])

    # Despejamos el vector de traslación t resolviendo K t = p b 
    t3 = p * m34
    t2 = (p * m24 - v0 * t3) / av
    t1 = (p * m14 - gamma * t2 - u0 * t3) / au
    t = np.array([t1, t2, t3])

   # Si se modificó K, recalculamos R proyectando C = p * K^(-1) * A mediante SVD
    if fuerza_k12_0 or fuerza_au_av_iguales:
        A_escalada = p * M[:, :3]
        C = mult_mat(inversa(K), A_escalada)
        U, _, V = svd_reducida(C)
        R = mult_mat(U, traspuesta(V))

    return R, t, K


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


def calcularMatrizK2D(A):
    """Recupera la matriz intrínseca de la calibración coplanar.

    Parámetros:
        A: matriz de tamaño (2*N, 6) construida por armarA, con al menos
            tres homografías que aporten restricciones independientes.
    Retorna:
        K de tamaño (3, 3), triangular superior y normalizada con K[2,2]=1,
        o None si ninguno de los signos de B permite Cholesky.

    Resuelve el sistema homogéneo Ab=0 por el menor autovalor de A.T A,
    elige el signo definido positivo de B y usa B=L L.T, K=(L.T)^-1.
    """

    S, D = diagRH(mult_mat(A.T, A))
    minimo = 0
    for i in range(1, 6):
        if D[i, i] < D[minimo, minimo]:
            minimo = i
    b = S[:, minimo]
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    # Solo uno de los signos puede dar una matriz definida positiva.
    L = calculaCholesky(B, atol=1e-15)
    if L is None:
        L = calculaCholesky(-B, atol=1e-15)
    if L is None:
        return None
    K = inversa(L.T)
    return K / K[2, 2]


def productoVectorial(a, b):
    """Calcula el producto vectorial de dos vectores de tres componentes.

    Parámetros:
        a, b: vectores de tamaño (3,).
    Retorna:
        Vector de tamaño (3,) correspondiente a a x b.
    """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def obtenerPose2D(Hj, K):
    """Recupera la pose del tablero a partir de su homografía.

    Parámetros:
        Hj: homografía de tamaño (3, 3), del plano del tablero a la imagen.
        K: matriz intrínseca de tamaño (3, 3).
    Retorna:
        R de tamaño (3, 3) y t de tamaño (3,), con la convención
        p_cámara=R p_tablero+t. Las unidades de t son las del tablero.
        Se supone una vista no degenerada con el origen delante de la cámara.
    """
    # Deshacemos los parámetros intrínsecos para recuperar las columnas de pose.
    columnas = mult_mat(inversa(K), Hj)
    escala = 1 / norma(columnas[:, 0], 2)
    if columnas[2, 2] < 0:
        escala = -escala
    r1 = escala * columnas[:, 0]
    r2 = escala * columnas[:, 1]
    t = escala * columnas[:, 2]

    # Los datos reales no producen columnas perfectamente ortonormales.
    r1 = r1 / norma(r1, 2)
    r2 = r2 - producto_interno(r1, r2) * r1
    r2 = r2 / norma(r2, 2)
    r3 = productoVectorial(r1, r2)
    R = np.array([r1, r2, r3]).T
    return R, t

def calculoLambdaMu(I0, J0):
    """Calcula los escalares lambda y mu para el caso coplanar.

    Parámetros:
        I0: vector de tamaño (3,). Solución POS para I (A^+ * b_u)
        J0: vector de tamaño (3,). Solución POS para J (A^+ * b_v)

    Retorna:
        Los dos pares de soluciones posibles (lambda1, mu1), (lambda2, mu2)
    """

    norma_I0 = norma(I0, 2)
    norma_J0 = norma(J0, 2)
    producto_I0_J0 = producto_interno(I0, J0)

    # Calculo la parte real e imaginaria de C^2 y creo el vector
    a = norma_J0 ** 2 - norma_I0 ** 2
    b = -2 * producto_I0_J0
    C_cuadrado = np.array([a, b])
    norma_C_cuadrado = norma(C_cuadrado, 2)

    # Calculo lamda y mu
    res_lambda = np.sqrt((norma_C_cuadrado + a) / 2.0)
    signo_b = 1 if b >= 0 else -1
    res_mu = signo_b * np.sqrt((norma_C_cuadrado - a) / 2.0)

    return (res_lambda, res_mu), (-res_lambda, -res_mu)

def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    """Recupera la pose (i, j, k, Z0, R, t) correspondiente a una solución (Is, Js) de POS.

    Parámetros:
        Is, Js : vectores de tamaño (3,)
            Solución completa I = I0 + lambda*n, J = J0 + mu*n del paso POS.
        fx : float
            Distancia focal en píxeles.
        m0 : vector de tamaño (2,)
            Imagen del punto de referencia M0, centrada y expresada en un
            modelo con focal fx en ambos ejes. aplicarPOSIT transforma antes
            los píxeles con K inversa si las focales son distintas.
        M0 : vector de tamaño (3,)
            Punto de referencia del objeto. La pose que devuelve esta función
            usa vectores relativos a M0: t es su posición en la cámara.
            El cambio al origen original se realiza en aplicarPOSIT.

    Retorna:
        i, j, k : vectores unitarios de tamaño (3,), ejes de la cámara en el sistema del objeto.
        Z0 : float, profundidad del punto de referencia.
        R : matriz de tamaño (3, 3), con filas i, j, k.
        t : vector de tamaño (3,), t = (Z0/f) * (x0, y0, f).
    """

    norma_I = norma(Is, 2)
    norma_J = norma(Js, 2)

    # Recuperamos la profundidad del punto de referencia
    Z0 = fx / norma_I

    # Normalizamos I y J y calculamos el tercer eje
    i = Is / norma_I
    j = Js / norma_J
    k = productoVectorial(i, j)

    R = np.array([i, j, k])

    # Ahora calculamos los puntos de referencia y el vector de traslacion
    x0, y0 = m0[0], m0[1]

    X0 = Z0 * x0/ fx
    Y0 = Z0 * y0/ fx

    t = np.array([X0, Y0, Z0])

    return i, j, k, Z0, R, t

def evaluarPose(R, t, obj_vecs, img_vecs, K):
    """Evalúa una pose reproyectando los puntos del objeto sobre la imagen.

    Parámetros:
        R : matriz (3, 3) de rotación (filas i, j, k).
        t : vector (3,) de traslación, posición de M0 en coordenadas de cámara.
        obj_vecs : arreglo (n, 3)
            Puntos del objeto en el sistema del objeto, con M0 en el origen
            (es decir, los vectores a_i = M0Mi).
        img_vecs : arreglo (n, 2)
            Posiciones (u, v) observadas en la imagen, en píxeles sin centrar.
        K : matriz (3, 3) de parámetros intrínsecos.

    Retorna:
        cantidad_validas : int
            Cantidad de puntos que quedan delante de la cámara (Z > 0).
        rms : float
            Raíz del error cuadrático medio (en píxeles) entre las proyecciones
            y img_vecs, calculado solo sobre los puntos válidos.
            Es infinito si ningún punto es válido.
    """
    cantidad_validas = 0
    suma_errores = 0.0

    for i in range(obj_vecs.shape[0]):
        # Pasamos a coordenadas de la camara
        vecs_camara = mult_mat_vector(R, obj_vecs[i]) + t

        # Salteamos los puntos que quedan detras de la camara
        if vecs_camara[2] <= 0:
            continue

        proyeccion = mult_mat_vector(K, vecs_camara)
        u = proyeccion[0] / proyeccion[2]
        v = proyeccion[1] / proyeccion[2]

        suma_errores += (u - img_vecs[i, 0]) ** 2 + (v - img_vecs[i, 1]) ** 2
        cantidad_validas += 1

    if cantidad_validas == 0:
        return 0, float('inf')

    rms = np.sqrt(suma_errores / cantidad_validas)
    return cantidad_validas, rms


def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    """Estima la pose de puntos coplanares mediante la iteración POSIT.

    Parámetros:
        obj_vecs: arreglo (n, 3) con los puntos coplanares del objeto, en una
            misma unidad y en correspondencia con las filas de img_vecs.
            El primer punto se usa como referencia M0.
        img_vecs: arreglo (n, 2) con sus píxeles observados, sin centrar.
        K: matriz intrínseca (3, 3), con K[2,2]=1 y focales positivas.
        MaxIt: máximo de pasos POS; debe ser un entero positivo.
        atol: reducción mínima del RMS, en píxeles, para seguir iterando.
    Retorna:
        R (3, 3) y t (3,) de la mejor pose visitada, con la convención
        p_cámara=R p_objeto+t. t tiene las unidades de obj_vecs.
        Lanza ValueError si no aparece una pose con todos los puntos delante
        de la cámara. Se requieren puntos coplanares no colineales.
    """
    puntos = np.asarray(obj_vecs, dtype=float)
    pixeles = np.asarray(img_vecs, dtype=float)
    if MaxIt < 1:
        raise ValueError('MaxIt debe ser positivo.')

    # Expresamos el objeto respecto del primer punto: A tiene rango dos.
    M0 = puntos[0]
    A = puntos - M0
    cantidad = A.shape[0]

    # Llevamos los píxeles a una cámara virtual de focal f en ambos ejes.
    # Así se contemplan las dos focales y el sesgo de la K recibida.
    f = K[0, 0]
    K_inversa = inversa(K)
    centrados = np.zeros((cantidad, 2))
    for indice in range(cantidad):
        u, v = pixeles[indice]
        rayo = mult_mat_vector(K_inversa, np.array([u, v, 1.0]))
        centrados[indice] = f * rayo[:2] / rayo[2]
    m0 = centrados[0]

    # Dos vectores independientes del plano determinan su normal unitaria.
    # Elegimos el par con mayor producto vectorial para evitar pares casi paralelos.
    mayor_norma = 0.0
    for fila in range(cantidad):
        for otra in range(fila):
            candidata = productoVectorial(A[fila], A[otra])
            longitud = norma(candidata, 2)
            if longitud > mayor_norma:
                mayor_norma = longitud
                normal = candidata / longitud
    if mayor_norma == 0:
        raise ValueError('Los puntos del objeto no deben ser colineales.')

    # Para rango dos, A^+=(A.T A+n n.T)^-1 A.T. El término n n.T
    # solo hace invertible la dirección normal, en la que A.T b vale cero;
    # por eso conserva la solución de mínimos cuadrados de norma mínima.
    columna_normal = normal.reshape((3, 1))
    matriz = mult_mat(A.T, A) + mult_mat(columna_normal, columna_normal.T)
    pseudoinversa = mult_mat(inversa(matriz), A.T)

    # La aproximación ortográfica inicial supone iguales profundidades.
    epsilon = np.zeros(cantidad)
    error_anterior = None
    mejor_error = float('inf')
    mejor_R, mejor_t = None, None
    for iteracion in range(MaxIt):
        b_u = centrados[:, 0] * (1 + epsilon) - m0[0]
        b_v = centrados[:, 1] * (1 + epsilon) - m0[1]
        I0 = mult_mat_vector(pseudoinversa, b_u)
        J0 = mult_mat_vector(pseudoinversa, b_v)

        # Examinamos ambas soluciones coplanares y rechazamos cualquiera
        # que deje al menos un punto detrás de la cámara o con Z=0.
        error_actual = float('inf')
        pose_actual = None
        for lam, mu in calculoLambdaMu(I0, J0):
            Is = I0 + lam * normal
            Js = J0 + mu * normal
            i, j, k, Z0, R, t_referencia = calcularSalidaPOSIT(Is, Js, f, m0, M0)
            validas, error = evaluarPose(R, t_referencia, A, pixeles, K)
            if validas == cantidad and error < error_actual:
                error_actual = error
                pose_actual = (k, Z0, R, t_referencia)
        if pose_actual is None:
            raise ValueError('Ninguna solución POSIT deja todos los puntos delante de la cámara.')

        k, Z0, R, t_referencia = pose_actual
        if error_actual < mejor_error:
            mejor_error = error_actual
            mejor_R = R
            # Volvemos al origen original del objeto, si M0 no era cero.
            mejor_t = t_referencia - mult_mat_vector(R, M0)

        # Si el RMS no mejora en al menos atol, conservamos la mejor pose.
        if error_anterior is not None and error_anterior - error_actual < atol:
            break
        error_anterior = error_actual
        epsilon = mult_mat_vector(A, k) / Z0

    return mejor_R, mejor_t
