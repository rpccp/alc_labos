import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Ejercicio 1
# ----------------------------

def ejercicio_1():
    x = 0.1 + 0.1 + 0.1
    y = 0.3

    print(x == y) # False

    print(format(x, '.20f')) # 0.30000000000000004441
    print(format(y, '.20f')) # 0.29999999999999998890

    return

#    ----------------------------
#    Ejercicio 2
#    ----------------------------

def ejercicio_2():

    # a = 1.0
    # while a != 0.1:
    #     print(a)
    #     a = a - 0.1
    # print('fin')

# NO ANDA POR ERROR DE REPRESENTACION

    a = 1.0
    eps = np.finfo('float').eps
    while a < 0.1 - eps or a > 0.1 + eps:
        a = a - 0.1
        print(a)
    print('fin')

    return

# ----------------------------
# Ejercicio 3
# ----------------------------

def ejercicio_3():

    x = 0.3 + 0.25
    y = 0.3 - 0.25

    print(format(0.3, '.20f'))
    print(format(0.25, '.20f'))

    print(format(0.55, '.20f'))
    print(format(0.05, '.20f'))

    print(format(x, '.20f'))
    print(format(y, '.20f'))


# No. En el caso de la suma si obtenemos el resultado esperado, sin embargo, en el caso de la resta no.
# Esto ocurre porque porque 0.25 tiene una representación binaria exacta, mientras que 0.3 no.
# 0.3 se almacena como una aproximación ligeramente menor. Al sumar, el redondeo final compensa esa diferencia y produce la misma representación que 0.55.
# En cambio, al restar, el pequeño error de 0.3 queda expuesto.

# (0.25) base 10 = (0.01) base 2

#      |  S   Exp           Mantisa
# 0.25 |  0 01111101 00000000000000000000000


# (0.3) base 10 = (0.010011001100110...) base 2
# Aparece una expansion periodica infinita, por lo tanto no se puede escribir exactamente con una mantisa finita.

    return

# ----------------------------
# Ejercicio 4
# ---------------------------

# np.sqrt(2)**2-2 da 4.440892098500626e-16

def ejercicio_4():
    

    x = np.linspace(0, 5e-8, 100)

    # Expresion a: tenemos cancelacion al restar dos numeros muy cercanos.
    y_a = np.sqrt(2*x**2 + 1) - 1

    # Expresion b: es algebraicamente equivalente, porque
    #
    # sqrt(2*x**2 + 1) - 1
    # = ((sqrt(2*x**2 + 1) - 1) * (sqrt(2*x**2 + 1) + 1))
    #   / (sqrt(2*x**2 + 1) + 1)
    # = (2*x**2) / (sqrt(2*x**2 + 1) + 1).
    y_b = (2*x**2) / (np.sqrt(2*x**2 + 1) + 1)

    plt.plot(x, y_a, label="Expresion a")
    plt.plot(x, y_b, label="Expresion b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparacion de expresiones matematicamente equivalentes")
    plt.legend()
    plt.grid()
    plt.show()

    # La expresion b resiste mejor la perdida de cifras significativas porque
    # evita restar sqrt(2*x**2 + 1), que es muy cercano a 1, menos 1.

# ----------------------------
# Ejercicio 5
# ----------------------------

def ejercicio_5():

    raiz_de_dos = np.sqrt(2)
    sucesion = [raiz_de_dos]

    # Algebraicamente, si x_n = sqrt(2), entonces
    # x_(n+1) = x_n**2 / sqrt(2) = 2 / sqrt(2) = sqrt(2).
    # La sucesion exacta es constante y su limite es sqrt(2).
    with np.errstate(over="ignore"):
        for _ in range(1, 100):
            x_anterior = sucesion[-1]
            sucesion.append(x_anterior*x_anterior / raiz_de_dos)


    primer_termino_desestabilizado = -1 
    i = 0
    while primer_termino_desestabilizado == -1 and i < len(sucesion):
        if not np.isfinite(sucesion[i]): primer_termino_desestabilizado = i
        i += 1


    print("Limite algebraico: sqrt(2) =", raiz_de_dos)
    print("La sucesion se desestabiliza en el punto:", primer_termino_desestabilizado)

    plt.semilogy(range(1, 101), sucesion, label="Sucesion calculada")
    plt.axhline(raiz_de_dos, color="red", linestyle="--", label="sqrt(2) exacto")
    plt.xlabel("i")
    plt.ylabel("x_i")
    plt.title("Ejercicio 5: acumulacion del error")
    plt.legend()
    plt.grid()
    plt.show()

    return sucesion


# ----------------------------
# Ejercicio 6
# ----------------------------

def suma_armonica(
    limite,
    tipo: type[np.float32] | type[np.float64] = np.float64,
    orden_inverso: bool = False,
) -> np.float32 | np.float64:
# Suma 1/i secuencialmente usando el tipo de flotante indicado.
    suma = tipo(0)
    indices = (
        range(limite, 0, -1)
        if orden_inverso
        else range(1, limite + 1)
    )

    for i in indices:
        suma = tipo(suma + tipo(1/i))

    return suma


def ejercicio_6(ns=(6, 7)):
    import math

    resultados = {}

    # Estas sumas son deliberadamente secuenciales: cambiar el orden o usar
    # np.sum cambia los redondeos que el ejercicio busca observar.
    for n in ns:
        for limite in (10**n, 5*10**n):
            for tipo in (np.float32, np.float64):
                suma = suma_armonica(limite, tipo)
                resultados[(n, limite, tipo.__name__)] = suma
                print(f"{tipo.__name__}, limite={limite}: {suma}")

    termino = np.float32(1/(2*10**7))
    print("1/(2*10**7) en float32 =", termino)

    # El termino no es cero, pero al sumarlo a una suma acumulada cercana a 16
    # es menor que la separacion entre flotantes de esa magnitud. Por eso la
    # operacion suma + termino devuelve nuevamente suma: el termino se pierde.
    suma_referencia = suma_armonica(5*10**6, np.float32)
    print(
        "¿La suma cambia al agregar 1/10**7?",
        suma_referencia + np.float32(1/10**7) != suma_referencia,
    )

    n_mayor = max(ns)
    suma_inversa = suma_armonica(2*10**n_mayor, np.float32, orden_inverso=True)
    print(f"Suma inversa hasta {2*10**n_mayor}:", suma_inversa)

    # Al sumar de menor a mayor magnitud, los terminos pequenos se acumulan
    # entre si antes de agregarse a los grandes y se pierden menos contribuciones.

    aproximacion_e = sum(1/math.factorial(n) for n in range(11))
    print("Aproximacion de e:", aproximacion_e)
    print("np.e:", np.e)
    print("Error absoluto:", error(np.e, aproximacion_e))

    return resultados


# ----------------------------
# Ejercicio 7
# ----------------------------

def ejercicio_7():
    A = np.array([
        [4, 2, 1],
        [2, 7, 9],
        [0, 5, 22/3],
    ], dtype=float)

    L = np.array([
        [1, 0, 0],
        [0.5, 1, 0],
        [0, 5/6, 1],
    ], dtype=float)

    U = np.array([
        [4, 2, 1],
        [0, 6, 8.5],
        [0, 0, 0.25],
    ], dtype=float)

    producto = L @ U
    print("A =\n", A)
    print("L @ U =\n", producto)
    print("¿A y L @ U son iguales?", matricesIguales(A, producto))

    return A, L, U


# ----------------------------
# Ejercicio 8
# ----------------------------

def esSimetrica(A):
    return matricesIguales(A, A.T)


def ejercicio_8():
    generador = np.random.default_rng(0)
    A = generador.random((4, 4))

    casos = {
        "A.T @ A": A.T @ A,
        "A.T @ ((A*0.25)/0.25)": A.T @ ((A*0.25)/0.25),
        "A.T @ ((A*0.2)/0.2)": A.T @ ((A*0.2)/0.2),
    }

    resultados = {}
    for nombre, matriz in casos.items():
        resultados[nombre] = esSimetrica(matriz)
        print(nombre, "es simetrica:", resultados[nombre])

    # Los tres casos son simetricos matematicamente. Con 0.2 pueden aparecer
    # diferencias de redondeo porque 0.2 no tiene representacion binaria finita.
    return resultados

# -----------------------------------------
# MODULO ALC
# -----------------------------------------
def error(x, y):
    return abs(x - np.float64(y))

# X != 0
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


if __name__ == "__main__":
    # ejercicio_1()
    # ejercicio_2()
    # ejercicio_3()
    # ejercicio_4()
    ejercicio_5()
    # ejercicio_7()
    # ejercicio_8()