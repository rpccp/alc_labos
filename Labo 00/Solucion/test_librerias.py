import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.show = lambda: None

import librerias as lib


class TestLibrerias(unittest.TestCase):
    def assertMatrizIgual(self, obtenida, esperada, places=None):
        self.assertEqual(obtenida.shape, esperada.shape)

        for i in range(len(obtenida)):
            for j in range(len(obtenida[0])):
                if places is None:
                    self.assertEqual(obtenida[i][j], esperada[i][j])
                else:
                    self.assertAlmostEqual(obtenida[i][j], esperada[i][j], places=places)

    def test_ejercicio_01_es_cuadrada(self):
        self.assertTrue(lib.esCuadrada(np.array([[1, 2], [3, 4]])))
        self.assertFalse(lib.esCuadrada(np.array([[1, 2, 3], [4, 5, 6]])))

    def test_ejercicio_02_triangular_superior_sin_diagonal(self):
        matriz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        esperada = np.array([[0, 2, 3], [0, 0, 6], [0, 0, 0]])
        self.assertMatrizIgual(lib.triangSup(matriz), esperada)
        with self.assertRaises(ValueError):
            lib.triangSup(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_ejercicio_03_triangular_inferior_sin_diagonal(self):
        matriz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        esperada = np.array([[0, 0, 0], [4, 0, 0], [7, 8, 0]])
        self.assertMatrizIgual(lib.triangInf(matriz), esperada)
        with self.assertRaises(ValueError):
            lib.triangInf(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_ejercicio_04_diagonal(self):
        matriz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        esperada = np.array([[1, 0, 0], [0, 5, 0], [0, 0, 9]])
        self.assertMatrizIgual(lib.diagonal(matriz), esperada)
        with self.assertRaises(ValueError):
            lib.diagonal(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_ejercicio_05_traza(self):
        self.assertEqual(lib.traza(np.array([[1, 2], [3, 4]])), 5)
        with self.assertRaises(ValueError):
            lib.traza(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_ejercicio_06_traspuesta_rectangular(self):
        matriz = np.array([[1, 2, 3], [4, 5, 6]])
        esperada = np.array([[1, 4], [2, 5], [3, 6]])
        self.assertMatrizIgual(lib.traspuesta(matriz), esperada)

    def test_ejercicio_07_simetrica(self):
        self.assertTrue(lib.esSimetrica(np.array([[1, 2], [2, 3]])))
        self.assertFalse(lib.esSimetrica(np.array([[1, 2], [3, 4]])))

    def test_ejercicio_08_calcular_ax(self):
        matriz = np.array([[1, 2], [3, 4], [5, 6]])
        vector = np.array([[2], [3]])
        esperada = np.array([[8], [18], [28]])
        self.assertMatrizIgual(lib.calcularAx(matriz, vector), esperada)
        with self.assertRaises(ValueError):
            lib.calcularAx(matriz, np.array([[1], [2], [3]]))
        with self.assertRaises(ValueError):
            lib.calcularAx(matriz, np.array([[1, 2], [3, 4]]))

    def test_ejercicio_09_intercambiar_filas_in_place(self):
        matriz = np.array([[1, 2], [3, 4], [5, 6]])
        resultado = lib.intercambiarFilas(matriz, 1, 3)
        self.assertIs(resultado, matriz)
        self.assertMatrizIgual(matriz, np.array([[5, 6], [3, 4], [1, 2]]))

    def test_ejercicio_10_sumar_fila_multiplo_in_place(self):
        matriz = np.array([[1, 2, 3], [4, 5, 6]])
        resultado = lib.sumar_fila_multiplo(matriz, 1, 2, 2)
        self.assertIs(resultado, matriz)
        self.assertMatrizIgual(matriz, np.array([[9, 12, 15], [4, 5, 6]]))

    def test_ejercicio_11_dominancia_diagonal_estricta(self):
        dominante = np.array([[4, 1, 1], [1, 4, 1], [1, 1, 4]])
        no_dominante = np.array([[1, 2], [1, 1]])
        igualdad = np.array([[1, 1], [0, 1]])
        self.assertTrue(lib.esDiagonalmenteDominante(dominante))
        self.assertFalse(lib.esDiagonalmenteDominante(no_dominante))
        self.assertFalse(lib.esDiagonalmenteDominante(igualdad))

    def test_ejercicio_12_matriz_circulante(self):
        vector = np.array([[1], [2], [3]])
        esperada = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
        self.assertMatrizIgual(lib.matrizCirculante(vector), esperada)

    def test_ejercicio_13_matriz_vandermonde(self):
        vector = np.array([[1], [2], [3]])
        esperada = np.array([[1, 1, 1], [1, 2, 3], [1, 4, 9]])
        self.assertMatrizIgual(lib.matrizVandermonde(vector), esperada)

    def test_ejercicio_14_numero_aureo(self):
        self.assertAlmostEqual(lib.numeroAureo(10), 89 / 55, places=12)

    def test_ejercicio_15_matriz_fibonacci(self):
        esperada = np.array([
            [0, 1, 1, 2],
            [1, 1, 2, 3],
            [1, 2, 3, 5],
            [2, 3, 5, 8]
        ])
        self.assertMatrizIgual(lib.matrizFibonacci(4), esperada)

    def test_ejercicio_16_matriz_hilbert(self):
        esperada = np.array([
            [1, 1 / 2, 1 / 3],
            [1 / 2, 1 / 3, 1 / 4],
            [1 / 3, 1 / 4, 1 / 5]
        ])
        self.assertMatrizIgual(lib.matrizHilbert(3), esperada, places=12)

    def test_ejercicio_17_polinomios(self):
        self.assertEqual(len(lib.puntos), 100)
        self.assertAlmostEqual(lib.puntos[0][0], -1, places=12)
        self.assertAlmostEqual(lib.puntos[99][0], 1, places=12)

        paso = 2 / 99
        for i in range(1, len(lib.puntos)):
            self.assertAlmostEqual(lib.puntos[i][0] - lib.puntos[i - 1][0], paso, places=12)

        p1 = lib.valores_polinomios["x^5 - x^4 + x^3 - x^2 + x - 1"]
        p2 = lib.valores_polinomios["x^2 + 3"]
        p3 = lib.valores_polinomios["x^10 - 2"]
        self.assertEqual(len(p1), 100)
        self.assertEqual(len(p2), 100)
        self.assertEqual(len(p3), 100)
        self.assertAlmostEqual(p1[0], -6, places=12)
        self.assertAlmostEqual(p1[99], 0, places=12)
        self.assertAlmostEqual(p2[0], 4, places=12)
        self.assertAlmostEqual(p2[99], 4, places=12)
        self.assertAlmostEqual(p3[0], -1, places=12)
        self.assertAlmostEqual(p3[99], -1, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
