"""Laboratorio 02 - Transformaciones lineales.

Resolución de los ejercicios 1 a 6. Al ejecutar este archivo se imprimen las
respuestas analíticas y se abren las visualizaciones pedidas en la consigna.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pointsGrid(esquinas):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 46),
                        np.linspace(esquinas[0,1], esquinas[1,1], 10))
 
    [w2, z2] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 10),
                        np.linspace(esquinas[0,1], esquinas[1,1], 46))
 
    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
 
    return wz


def proyectarPts(T, wz):
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = None
    xy = T @ wz
    return xy


def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)
 
    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             
 
    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')


def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)


def main():
    print('Ejecutar el programa')
    # generar el tipo de transformacion dando valores a la matriz T
    T = pd.read_csv('C:/Users/rgasp/Desktop/ALC/Recursos_Laboratorio 02_TransformacionesLineales_20260829/T.csv', header=None).values
    corners = np.array([[0,0],[100,100]])
    # corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    wz = pointsGrid(corners)
    vistform(T, wz, 'Deformar coordenadas')
    plt.show()



# -----------------
# # MODULO ALC
# -----------------

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


# ----------------------------
# Respuestas ejercicios
# -----------------------------

# EJERCICIO 1
# La t.l. tiene matriz
#     T = [[1/2, 0],
#           [0, 1/2]].
# Y
#     T^(-1) = [[2, 0], 
#               [0, 2]].
# T achica todas las longitudes a la mitad, la inversa las duplica.

# EJERCICIO 2
# Para (x,y)=(aw,bz),
#     T = [[a, 0], 
#           [0, b]],
#     T^(-1) = [[1/a, 0], 
#               [0, 1/b]],  con a,b != 0.
# Si a=2 y b=3: e1 pasa a (2,0), e2 a (0,3), el punto (1,1) pasa
# a (2,3) y una circunferencia pasa a una elipse.

# EJERCICIO 3
# (a) 
#     T = [[1, c],
#           [d, 1]],
#     T^(-1) = 1/(1-cd) [[1, -c], 
#                       [-d, 1]],  si cd != 1.

# EJERCICIO 4
# (a) Las columnas son las imágenes de e1 y e2:
#     R(theta) = [[cos(theta), -sin(theta)],
#                 [sin(theta),  cos(theta)]].
# a(theta)=cos(theta), b(theta)=-sin(theta),
# c(theta)=sin(theta) y d(theta)=cos(theta).

# (b) Para 0, pi/2, pi, 3pi/2 y 2pi se obtienen giros de 0, 90, 180, 270 y
# 360 grados. Las rotaciones se distinguen de las deformaciones de cizalla, ya que
# las rotaciones conservan longitudes, ángulos y áreas.

# EJERCICIO 5
# Primero se aplica R(pi/4), luego D=diag(2,3) y al final R(-pi/4).
# La matriz compuesta es
#     Q = R(-pi/4) D R(pi/4) = [[5/2, 1/2],
#                               [1/2, 5/2]].
# La circunferencia unidad pasa a una elipse, y sus semiejes miden 2 y 3.
# Sus ejes principales quedan orientados según las direcciones rotadas.

# EJERCICIO 6
# En coordenadas homogéneas, (x,y)=A(w,z)+b se representa mediante
#     [[a11, a12, b1], [a21, a22, b2], [0, 0, 1]].
# Las operaciones pedidas son:
#   Escalar:  [[sx, 0, 0], [0, sy, 0], [0, 0, 1]].
#   Trasladar:[[1, 0, b1], [0, 1, b2], [0, 0, 1]].
#   Rotar:    [[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]].
#   Deformar: [[1, c, 0], [d, 1, 0], [0, 0, 1]].
# La función proyectarPts acepta estas matrices 3x3. La función afin construye
# una composición que primero rota, luego escala y por último traslada.


# -----------------------------------------------------------------------------
# Visualizaciones
# -----------------------------------------------------------------------------

def _limites_cuadrados(*conjuntos):
    """Calcula límites cuadrados comunes para varios conjuntos de puntos."""
    todos = np.concatenate(conjuntos, axis=1)
    minimo = np.min(todos, axis=1)
    maximo = np.max(todos, axis=1)
    centro = (minimo + maximo) / 2
    radio = max(np.max(maximo - minimo) / 2, 0.5) * 1.12
    return ((centro[0] - radio, centro[0] + radio),
            (centro[1] - radio, centro[1] + radio))


def _configurar_ejes(ax, titulo, limites=None):
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal")
    ax.set_title(titulo)
    if limites is not None:
        ax.set_xlim(limites[0])
        ax.set_ylim(limites[1])


def cizalla(horizontal=0, vertical=0):
    """Devuelve [[1, horizontal], [vertical, 1]]."""
    return np.array([[1, horizontal], [vertical, 1]], dtype=float)


def rotacion_reescalada(theta, s):
    """Rota theta, escala según s y deshace la rotacion inicial."""
    return rota(-theta) @ escala(s) @ rota(theta)


def graficar_ejercicios():
    """Construye y devuelve las siete figuras correspondientes a los ejercicios."""
    figuras = []
    grilla = pointsGrid(np.array([[-1, -1], [1, 1]]))

    # Ejercicio 1: escala uniforme y su inversa.
    T1 = escala([0.5, 0.5])
    fig1, axes = plt.subplots(2, 2, figsize=(8, 8))
    for fila, (matriz, nombre) in enumerate(
        ((T1, "T: reduce a la mitad"), (np.linalg.inv(T1), "T^(-1): duplica"))
    ):
        salida = proyectarPts(matriz, grilla)
        limites = _limites_cuadrados(grilla, salida)
        axes[fila, 0].plot(grilla[0], grilla[1], ".", markersize=2)
        axes[fila, 1].plot(salida[0], salida[1], ".", markersize=2)
        _configurar_ejes(axes[fila, 0], f"Entrada para {nombre}", limites)
        _configurar_ejes(axes[fila, 1], f"Salida de {nombre}", limites)
    fig1.suptitle("Ejercicio 1 - Transformación directa e inversa")
    fig1.tight_layout()
    figuras.append(fig1)

    # Ejercicio 2: base canónica, punto arbitrario y circunferencia.
    angulos = np.linspace(0, 2 * np.pi, 361)
    circulo = np.vstack((np.cos(angulos), np.sin(angulos)))
    base_y_punto = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    T2 = escala([2, 3])
    circulo_t = proyectarPts(T2, circulo)
    puntos_t = proyectarPts(T2, base_y_punto)
    limites = _limites_cuadrados(circulo, circulo_t, base_y_punto, puntos_t)
    fig2, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, curva, puntos, nombre in (
        (axes[0], circulo, base_y_punto, "Antes"),
        (axes[1], circulo_t, puntos_t, "Después: a=2, b=3"),
    ):
        ax.plot(curva[0], curva[1], linewidth=2)
        ax.scatter(puntos[0], puntos[1], color=["C1", "C2", "C3"], zorder=3)
        for punto, etiqueta in zip(puntos.T, ("e1", "e2", "p=(1,1)")):
            ax.annotate(etiqueta, punto, xytext=(5, 5), textcoords="offset points")
        _configurar_ejes(ax, nombre, limites)
    fig2.suptitle("Ejercicio 2 - Reescalamiento")
    fig2.tight_layout()
    figuras.append(fig2)

    # Ejercicio 3b y 3c: evolución al variar el parámetro b.
    valores_b = np.linspace(0, 1, 5)
    fig3, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for ax, b in zip(axes, valores_b):
        salida = proyectarPts(cizalla(horizontal=b), grilla)
        ax.plot(salida[0], salida[1], ".", markersize=1.5)
        _configurar_ejes(ax, f"b={b:.2f}", ((-2.2, 2.2), (-2.2, 2.2)))
    fig3.suptitle("Ejercicio 3b - Cizalla horizontal")
    fig3.tight_layout()
    figuras.append(fig3)

    fig4, axes = plt.subplots(1, 5, figsize=(15, 3.8))
    for ax, b in zip(axes, valores_b):
        salida = proyectarPts(cizalla(horizontal=b, vertical=b), circulo)
        ax.plot(salida[0], salida[1], linewidth=2)
        _configurar_ejes(ax, f"b=c={b:.2f}", ((-2.2, 2.2), (-2.2, 2.2)))
    fig4.suptitle("Ejercicio 3c - Deformación simétrica")
    fig4.tight_layout(rect=(0, 0, 1, 0.9))
    figuras.append(fig4)

    # Ejercicio 4: rotaciones indicadas en la consigna.
    rotaciones = (0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi)
    referencia = np.array([[0.8], [0.3]])
    fig5, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for ax, theta in zip(axes, rotaciones):
        salida = proyectarPts(rota(theta), grilla)
        referencia_rotada = proyectarPts(rota(theta), referencia)
        ax.plot(salida[0], salida[1], ".", markersize=1.5)
        ax.scatter(
            referencia_rotada[0],
            referencia_rotada[1],
            color="C3",
            zorder=3,
        )
        _configurar_ejes(
            ax,
            f"theta={theta / np.pi:g} pi",
            ((-1.5, 1.5), (-1.5, 1.5)),
        )
    fig5.suptitle("Ejercicio 4 - Rotaciones")
    fig5.tight_layout()
    figuras.append(fig5)

    # Ejercicio 5: círculo transformado por R(-45) S R(45).
    T5 = rotacion_reescalada(np.pi / 4, [2, 3])
    circulo_t = proyectarPts(T5, circulo)
    limites = _limites_cuadrados(circulo, circulo_t)
    fig6, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(circulo[0], circulo[1], linewidth=2)
    axes[1].plot(circulo_t[0], circulo_t[1], linewidth=2)
    _configurar_ejes(axes[0], "Circunferencia original", limites)
    _configurar_ejes(axes[1], "Elipse transformada", limites)
    fig6.suptitle("Ejercicio 5 - Rotacion, escala y rotacion inversa")
    fig6.tight_layout()
    figuras.append(fig6)

    # Ejercicio 6: matrices afines 3x3 para las cuatro operaciones básicas.
    # matrices_afines = (
    #     (np.array([[1.5, 0, 0], [0, 0.5, 0], [0, 0, 1]]), "Escala"),
    #     (np.array([[1, 0, 1], [0, 1, 0.5], [0, 0, 1]]), "Traslación"),
    #     (afin(np.pi / 4, [1, 1], [0, 0]), "Rotacion"),
    #     (np.array([[1, 0.7, 0], [0, 1, 0], [0, 0, 1]]), "Cizalla"),
    # )
    # fig7, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    # for ax, (matriz, nombre) in zip(axes, matrices_afines):
    #     salida = proyectarPts(matriz, grilla)
    #     ax.plot(salida[0], salida[1], ".", markersize=1.5)
    #     _configurar_ejes(ax, nombre, ((-2.2, 2.2), (-2.2, 2.2)))
    # fig7.suptitle("Ejercicio 6 - Transformaciones afines en coordenadas homogéneas")
    # fig7.tight_layout()
    # figuras.append(fig7)

    return figuras


if __name__ == "__main__":
     main()

graficar_ejercicios()
plt.show()