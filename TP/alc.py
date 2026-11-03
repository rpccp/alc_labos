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

def resolverDLT3D(Q):
    """Resuelve el sistema Qm = 0 para encontrar la matriz de proyección M.

    Parámetros:
        Q : numpy.ndarray de tamaño (2*n, 12)
            Matriz de diseño del sistema obtenida a partir de las correspondencias 2D-3D.

    Retorna:
        M : numpy.ndarray de tamaño (3, 4)
            Matriz de proyección de la cámara calculada mediante cuadrados mínimos.
    """
    Qt = traspuesta(Q)
    QtQ = mult_mat(Qt, Q)
    
    # Calculamos autovectores (S) y autovalores (D) de QtQ.
    # Las columnas de S equivalen a la matriz V de la SVD de Q.
    S, D = diagRH(QtQ)
    
    # El vector que minimiza el error es el autovector asociado al menor autovalor.
   
    # Como diagRH los ordena de mayor a menor, extraemos la última columna.
    m = S[:, -1]
    
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
    if np.shape(M) != (3, 4):
        return None, None, None

    # Extraemos los primeros tres elementos de cada fila (qi) y el último elemento (mi4)
    q1 = M[0, :3]
    q2 = M[1, :3]
    q3 = M[2, :3]
    
    m14 = M[0, 3]
    m24 = M[1, 3]
    m34 = M[2, 3]

    # Calculamos el factor de escala p
    p = 1 / norma(q3, 2)

    # Despejamos la tercera fila de la matriz de rotación y el tercer componente de t
    r3 = p * q3
    t3 = p * m34

    # Calculamos v0 aprovechando la ortogonalidad con r3
    v0 = p * producto_interno(q2, r3)
    
    # Aislamos la segunda fila de rotación, calculamos av y normalizamos
    r2_sin_norma = (p * q2) - (v0 * r3)
    av = norma(r2_sin_norma, 2)
    r2 = r2_sin_norma / av
    
    # Despejamos el segundo componente de t
    t2 = (p * m24 - v0 * t3) / av

    # Calculamos u0 usando la ortogonalidad
    u0 = p * producto_interno(q1, r3)            
    
    # Verificamos si tenemos que forzar gamma (K12) = 0
    if fuerza_k12_0:
        gamma = 0.0
    else:
        gamma = p * producto_interno(q1, r2)          
    
    # Aislamos la primera fila de rotación
    r1_sin_normalizar = (p * q1) - (gamma * r2) - (u0 * r3)
    
    # Verificamos si tenemos que forzar au = av
    if fuerza_au_av_iguales:
        au = av
    else:
        au = norma(r1_sin_normalizar, 2)
        
    r1 = r1_sin_normalizar / au
    
    # Despejamos el primer componente de t
    t1 = (p * m14 - gamma * t2 - u0 * t3) / au

    # Armamos la matriz intrínseca K
    K = np.array([
        [au, gamma, u0],
        [0.0, av, v0],
        [0.0, 0.0, 1.0]
    ])

    # Armamos la matriz ortogonal de rotación R
    R = np.array([r1, r2, r3])
    
    # Armamos el vector de traslación t
    t = np.array([t1, t2, t3])

    return R, t, K


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


def calculaCholesky(A, atol=1e-15):
    """Calcula la factorización de Cholesky sin modificar A.

    Parámetros:
        A: matriz simétrica de tamaño (n, n).
        atol: tolerancia para considerar positivo cada pivote.
    Retorna:
        L triangular inferior tal que A=L L.T, o None si un pivote es
        menor o igual a atol.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    # Cada entrada usa solo columnas anteriores de L ya calculadas.
    for i in range(n):
        for j in range(i + 1):
            suma = 0.0
            for k in range(j):
                suma += L[i, k] * L[j, k]
            if i == j:
                pivote = A[i, i] - suma
                if pivote <= atol:
                    return None
                L[i, j] = pivote ** 0.5
            else:
                L[i, j] = (A[i, j] - suma) / L[j, j]
    return L


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
    L = calculaCholesky(B)
    if L is None:
        L = calculaCholesky(-B)
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
            Imagen del punto de referencia M0, ya centrada respecto del centro
            principal: (x0, y0) = (u0 - cu, v0 - cv).

    Retorna:
        i, j, k : vectores unitarios de tamaño (3,), ejes de la cámara en el sistema del objeto.
        Z0 : float, profundidad del punto de referencia.
        R : matriz de tamanño (3, 3), con filas i, j, k.
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
        vecs_camara = mult_mat_vector(R, obj_vecs) + t

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