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
