import numpy as np


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
    A = rota_y_escala(theta, s)

    # Convertimos b a vector columna
    b = np.reshape(b, (2, 1))

    # Le agregamos b como tercera columna a A
    A = np.hstack((A, b))

    # Le agregamos una nueva fila a A y devolvemos
    return np.vstack((A, [0, 0, 1]))
                 

def trans_afin(v, theta, s, b):
    v = np.reshape(v, (2, 1))

    # Le agregamos un 1 abajo
    v =np.vstack((v, [1]))

    # Calculamos el resultante de aplicar la transformacion afin
    w = afin(theta, s, b) @ v

    # Le quitamos la ultima fila
    w = np.delete(w, 2, 0)

    # Convertimos a w en vector fila y devolvemos
    return np.reshape(w, (1, 2))