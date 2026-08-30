import numpy as np
import matplotlib.pyplot as plt

a_matrix = np.array([[ 20,  1,  2, 3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11],
                    [12, 13, 14, 15]])

# Ejercicio 1
def esCuadrada(A: np.ndarray) -> bool:
    (cant_filas, cant_columnas) = A.shape
    return cant_filas == cant_columnas



# Ejercicio 2
def triangSup(A: np.ndarray) -> np.ndarray:
    if not esCuadrada(A):
        raise ValueError("La matriz debe ser cuadrada")

    (cant_filas, cant_columnas) = A.shape
    U: np.ndarray = A.copy()
    for i in range(cant_filas):
        for j in range(cant_columnas):
            if i >= j:
                U[i][j] = 0
    return U



# Ejercicio 3
def triangInf(A: np.ndarray) -> np.ndarray:
    if not esCuadrada(A):
        raise ValueError("La matriz debe ser cuadrada")
    
    (cant_filas, cant_columnas) = A.shape
    L: np.ndarray = A.copy()
    for i in range(cant_filas):
        for j in range(cant_columnas):
            if i <= j:
                L[i][j] = 0
    return L



# Ejercicio 4
def diagonal(A: np.ndarray) -> np.ndarray:
    if not esCuadrada(A):
        raise ValueError("La matriz debe ser cuadrada")

    (cant_filas, cant_columnas) = A.shape
    D: np.ndarray = A.copy()
    for i in range(cant_filas):
        for j in range(cant_columnas):
            if i != j:
                D[i][j] = 0
    return D



# Ejercicio 5
def traza(A: np.ndarray):
    if not esCuadrada(A):
        raise ValueError("La matriz debe ser cuadrada")

    suma: int = 0
    for i in range(len(A)):
        suma += A[i][i]
    return suma



# Ejercicio 6
def traspuesta(A: np.ndarray) -> np.ndarray:
    At: list[list] = []
    (cant_filas, cant_columnas) = A.shape

    for i in range(cant_columnas):
        nueva_fila: list = []

        for j in range(cant_filas):
            nueva_fila.append(A[j][i])

        At.append(nueva_fila)

    return np.array(At)



# Ejercicio 7
def esSimetrica(A: np.ndarray) -> bool:
    (cant_filas, cant_columnas) = A.shape

    for i in range(cant_filas):
        for j in range(cant_columnas):
            if i != j:
                if A[i][j] != A[j][i]:
                    return False

    return True



# Ejercicio 8
def calcularAx(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    (cant_filas_A, cant_columnas_A) = A.shape
    (cant_filas_x, cant_columnas_x) = x.shape

    if cant_columnas_A != cant_filas_x:
        raise ValueError("Las dimensiones de A y x no permiten su multiplicacion")

    if cant_columnas_x != 1:
        raise ValueError("x no es un vector")

    b: list[list] = []

    for i in range(cant_filas_A):
        nuevo_valor:int = 0

        for j in range(cant_columnas_A):
            nuevo_valor += A[i][j] * x[j][0]

        b.append([nuevo_valor])

    return np.array(b)



# Ejercicio 9
# Asumo que la primera fila es la fila 1
def intercambiarFilas(A: np.ndarray, i:int, j:int) -> np.ndarray:
    fila_i: int = i - 1
    fila_j: int = j - 1
    for k in range(len(A[0])):
        elem: int = A[fila_i][k]
        A[fila_i][k] = A[fila_j][k]
        A[fila_j][k] = elem

    return A



# Ejercicio 10
def sumar_fila_multiplo(A: np.ndarray, i: int, j: int, s: int) -> np.ndarray:
    fila_i: int = i - 1
    fila_j: int = j - 1
    for k in range(len(A[0])):
        A[fila_i][k] = A[fila_i][k] + (A[fila_j][k] * s)

    return A



# Ejercicio 11
def suma_abs_elementos(A: np.ndarray) -> int:
    suma: int = 0
    for i in range(len(A)):
        suma += abs(A[i])

    return suma



def esDiagonalmenteDominante(A: np.ndarray) -> bool:
    for i in range(len(A)):
        if abs(A[i][i]) <= (suma_abs_elementos(A[i]) - abs(A[i][i])):
            return False
        
    return True



# Ejercicio 12
def matrizCirculante(v:np.ndarray) -> np.ndarray:
    matriz_circulante: list[list] = []
    for i in range(len(v)):
        nueva_fila: list = []
        for j in range(len(v)):
            nueva_fila.append(v[(j - i) % len(v)][0])

        matriz_circulante.append(nueva_fila)

    return np.array(matriz_circulante)



# Ejercicio 13
def matrizVandermonde(v: np.ndarray) -> np.ndarray:
    V: list[list] = []
    for i in range(len(v)):
        nueva_fila: list = []
        for j in range(len(v)):
            nueva_fila.append(v[j][0] ** i)

        V.append(nueva_fila)

    return np.array(V)



# Ejercicio 14
def numeroAureo(n):
    F: np.ndarray = np.array([[0, 1],
                              [1, 1]])

    semilla: np.ndarray = np.array([[0], [1]])

    for i in range(n):
        nueva_semilla: list = []
        nueva_semilla.append([F[0][0] * semilla[0][0] + F[0][1] * semilla[1][0]])
        nueva_semilla.append([F[1][0] * semilla[0][0] + F[1][1] * semilla[1][0]])
        semilla = np.array(nueva_semilla)

    return semilla[1][0] / semilla[0][0]



# Ejercicio 15
def fibonacci(n: int) -> int:
    if n == 0: return 0
    if n == 1: return 1

    return fibonacci(n - 1) + fibonacci(n - 2)


def matrizFibonacci(n: int) -> np.ndarray:
    F: list[list[int]] = []
    for i in range(n):
        nueva_fila: list[int] = []
        for j in range(n):
            nueva_fila.append(fibonacci(i + j))

        F.append(nueva_fila)

    return np.array(F)



# Ejercicio 16
def matrizHilbert(n: int) -> np.ndarray:
    H: list[list] = []
    for i in range(n):
        nueva_fila: list = []
        for j in range(n):
            nueva_fila.append(1 / (i + j + 1))

        H.append(nueva_fila)

    return np.array(H)



# Ejercicio 17
puntos_lista: list[list[float]] = []

for i in range(100):
    punto: float = -1 + i * (2 / 99)
    puntos_lista.append([punto])

puntos: np.ndarray = np.array(puntos_lista)
polinomios: dict[str, np.ndarray] = {
    "x^5 - x^4 + x^3 - x^2 + x - 1": np.array([[-1], [1], [-1], [1], [-1], [1]]),
    "x^2 + 3": np.array([[3], [0], [1]]),
    "x^10 - 2": np.array([[-2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1]])
}

tam_bloque: int = 11
valores_polinomios: dict[str, np.ndarray] = {}

for nombre, coeficientes in polinomios.items():
    valores = []
    coeficientes_ajustados_lista: list[list[float]] = []

    for i in range(tam_bloque):
        if i < len(coeficientes):
            coeficientes_ajustados_lista.append([coeficientes[i][0]])
        else:
            coeficientes_ajustados_lista.append([0])

    coeficientes_ajustados: np.ndarray = np.array(coeficientes_ajustados_lista)

    for inicio in range(0, len(puntos), tam_bloque):
        bloque: np.ndarray = puntos[inicio:inicio + tam_bloque]

        if len(bloque) < tam_bloque:
            bloque = puntos[-tam_bloque:]

        vandermonde: np.ndarray = matrizVandermonde(bloque)
        valores_bloque: np.ndarray = calcularAx(traspuesta(vandermonde), coeficientes_ajustados)

        if inicio + tam_bloque > len(puntos):
            valores_bloque = valores_bloque[-(len(puntos) - inicio):]

        valores.extend(valores_bloque[:, 0])

    valores_polinomios[nombre] = np.array(valores)


for nombre, valores in valores_polinomios.items():
    plt.plot(puntos[:, 0], valores, label=nombre)

plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Ejercicio 17: valores de polinomios entre -1 y 1")
plt.grid()
plt.legend()
plt.show()



#Ejercicio 18
def filas_con_mayor_modulo_en_primer_columna(A: np.ndarray) -> list:
    filas_con_mayor_modulo: list = []
    mayor_modulo = 0
    for fila in A:
        if abs(fila[0]) > mayor_modulo:
            mayor_modulo = abs(fila[0])

    for i in range(A.shape[0]):
        if abs(A[i, 0]) == mayor_modulo:
            filas_con_mayor_modulo.append(i)

    return filas_con_mayor_modulo 


def row_echelon(M):
    """ 
        Retorna la Matriz Escalonada por Filas 
    """
    A = np.copy(M)
    if (issubclass(A.dtype.type, np.integer)):
        A = A.astype(float)
    # Si A no tiene filas o columnas, ya esta escalonada
    f, c = A.shape
    if f == 0 or c == 0:
        return A
    # buscamos primer elemento no nulo de la primera columna
    i = 0
    while i < f and (A[i,0] == 0 or (i not in filas_con_mayor_modulo_en_primer_columna(A))):
        i += 1
    if i == f:
        # si todos los elementos de la primera columna son ceros
        # escalonamos filas desde la segunda columna
        B = row_echelon(A[:,1:])
        # y volvemos a agregar la primera columna de zeros
        return np.block([A[:,:1], B])
    # intercambiamos filas i <-> 0, pues el primer elemento no nulo aparece en la fila i
    if i > 0:
        A[[0,i],:] = A[[i,0],:]
    # PASO DE TRIANGULACION GAUSSIANA:
    # a las filas subsiguientes les restamos un multiplo de la primera
    A[1:,:] -= (A[0,:] / A[0,0]) * A[1:,0:1]
    # escalonamos desde la segunda fila y segunda columna en adelante
    B = row_echelon(A[1:,1:])
    # reconstruimos la matriz por bloques adosando a B la primera fila 
    # y la primera columna (de ceros)
    return np.block([ [A[:1,:]], [ A[1:,:1], B] ])