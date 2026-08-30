
import numpy as np
import matplotlib.pyplot as plt
import main as labo
from main import proyectarPts, rota, escala, rota_y_escala, afin, trans_afin


# --- Tests extraídos del enunciado ---

print("\nIniciando serie de tests del Laboratorio 2...")

# Una matriz afín 3x3 debe transformar puntos 2D usando coordenadas homogéneas.
puntos = np.array([[0, 2, -1],
                   [0, 3, 4]])
traslacion = np.array([[1, 0, 5],
                       [0, 1, -2],
                       [0, 0, 1]])
assert(np.allclose(
    proyectarPts(traslacion, puntos),
    np.array([[5, 7, 4],
              [-2, 1, 2]])
))

# Una cizalla horizontal suma b veces la coordenada vertical a la horizontal.
assert(hasattr(labo, "cizalla"))
assert(np.allclose(
    labo.cizalla(0.4, 0),
    np.array([[1, 0.4],
              [0, 1]])
))

# Ejercicio 5: importa el orden, primero R(45), luego S y por último R(-45).
assert(hasattr(labo, "rotacion_reescalada"))
assert(np.allclose(
    labo.rotacion_reescalada(np.pi / 4, [2, 3]),
    np.array([[2.5, 0.5],
              [0.5, 2.5]])
))

# Tests para rota
assert(np.allclose(rota(0), np.eye(2)))
assert(np.allclose(rota(np.pi/2), np.array([[0, -1],[1, 0]])))
assert(np.allclose(rota(np.pi), np.array([[-1, 0],[0, -1]])))

# Tests para escala
assert(np.allclose(escala([2,3]), np.array([[2,0],[0,3]])))
assert(np.allclose(escala([1,1,1]), np.eye(3)))
assert(
    np.allclose(escala([0.5,0.25]), np.array([[0.5,0],[0,0.25]]))
)

# Tests para rota_y_escala
assert(
    np.allclose(rota_y_escala(0,[2,3]), np.array([[2,0],[0,3]]))
)
assert(np.allclose(
    rota_y_escala(np.pi/2,[1,1]), np.array([[0,-1],[1,0]])
))
assert(np.allclose(
    rota_y_escala(np.pi,[2,2]), np.array([[-2,0],[0,-2]])
))

# Tests para afin
assert(np.allclose(
    afin(0,[1,1],[1,2]),
    np.array([[1,0,1],
              [0,1,2],
              [0,0,1]])))
assert(np.allclose(afin(np.pi/2,[1,1],[0,0]),
    np.array([[0,-1,0],
              [1,0,0],
              [0,0,1]])))
assert(np.allclose(afin(0,[2,3],[1,1]),
    np.array([[2,0,1],
              [0,3,1],
              [0,0,1]])))

# Tests para trans_afina

assert(np.allclose(
    trans_afin(np.array([1,0]), np.pi/2,[1,1],[0,0]),
    np.array([0,1])
))
assert(np.allclose(
    trans_afin(np.array([1,1]), 0,[2,3],[0,0]),
    np.array([2,3])
))
assert(np.allclose(
    trans_afin(np.array([1,0]), np.pi/2,[3,2],[4,5]),
    np.array([4,7])
))
assert(trans_afin(np.array([1, 0]), 0, [1, 1], [0, 0]).shape == (2,))

# La entrega debe poder construir todas las demostraciones gráficas sin errores.
assert(hasattr(labo, "graficar_ejercicios"))
figuras = labo.graficar_ejercicios()
assert(len(figuras) == 7)

# El título general del ejercicio 3c no debe tapar los títulos de los casos.
figura_3c = figuras[3]
figura_3c.canvas.draw()
renderizador = figura_3c.canvas.get_renderer()
caja_titulo_general = figura_3c._suptitle.get_window_extent(renderizador)
assert(not any(
    caja_titulo_general.overlaps(ax.title.get_window_extent(renderizador))
    for ax in figura_3c.axes
))

# Las rotaciones deben incluir una referencia asimétrica que haga visible el giro.
assert(all(ax.collections for ax in figuras[4].axes))
plt.close("all")

print("¡Todos los tests del Laboratorio 2 han pasado exitosamente!")
