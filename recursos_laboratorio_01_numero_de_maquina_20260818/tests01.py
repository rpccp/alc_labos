"""Tests para las funciones del módulo ALC del Laboratorio 1.

Se pueden ejecutar directamente con:

    python tests01.py
"""

import unittest

import numpy as np

if __package__:
    from .lab1 import error, error_relativo, matricesIguales
else:
    from lab1 import error, error_relativo, matricesIguales


def sonIguales(x, y, atol=1e-08):
    """Auxiliar provisto por la cátedra."""
    return np.allclose(error(x, y), 0, atol=atol)


class TestsProvistosPorLaCatedra(unittest.TestCase):
    """Casos incluidos al final del enunciado."""

    def test_error(self):
        self.assertFalse(sonIguales(1, 1.1))
        self.assertTrue(sonIguales(1, 1 + np.finfo("float64").eps))
        self.assertFalse(sonIguales(1, 1 + np.finfo("float32").eps))
        self.assertFalse(
            sonIguales(
                np.float16(1),
                np.float16(1) + np.finfo("float32").eps,
            )
        )
        self.assertTrue(
            sonIguales(
                np.float16(1),
                np.float16(1) + np.finfo("float16").eps,
                atol=1e-3,
            )
        )

    def test_error_relativo(self):
        self.assertTrue(np.allclose(error_relativo(1, 1.1), 0.1))
        self.assertTrue(np.allclose(error_relativo(2, 1), 0.5))
        self.assertTrue(np.allclose(error_relativo(-1, -1), 0))
        self.assertTrue(np.allclose(error_relativo(1, -1), 2))

    def test_matrices_iguales(self):
        self.assertTrue(matricesIguales(np.diag([1, 1]), np.eye(2)))

        A = np.array([[1, 2], [3, 4]])
        self.assertTrue(matricesIguales(np.linalg.inv(A) @ A, np.eye(2)))

        self.assertFalse(matricesIguales(A.T, A))


class TestsAdicionalesError(unittest.TestCase):
    def test_numeros_iguales_tienen_error_cero(self):
        self.assertEqual(error(3.5, 3.5), 0)

    def test_error_siempre_es_no_negativo(self):
        self.assertGreaterEqual(error(2, 5), 0)
        self.assertGreaterEqual(error(-2, -5), 0)

    def test_error_con_numeros_negativos(self):
        self.assertTrue(np.allclose(error(-3, -2.5), 0.5))

    def test_error_con_distintos_tipos_float(self):
        resultado = error(np.float32(1.5), np.float16(1.25))
        self.assertTrue(np.allclose(resultado, 0.25))


class TestsAdicionalesErrorRelativo(unittest.TestCase):
    def test_resultado_con_valor_exacto_negativo(self):
        self.assertTrue(np.allclose(error_relativo(-2, -1), 0.5))

    def test_resultado_es_no_negativo(self):
        self.assertGreaterEqual(error_relativo(-4, 2), 0)

    def test_es_invariante_ante_cambio_de_escala(self):
        original = error_relativo(4, 3)
        escalado = error_relativo(400, 300)
        self.assertTrue(np.allclose(original, escalado))


class TestsAdicionalesMatricesIguales(unittest.TestCase):
    def test_misma_matriz_rectangular(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(matricesIguales(A, A.copy()))

    def test_dimensiones_distintas(self):
        A = np.zeros((2, 3))
        B = np.zeros((3, 2))
        self.assertFalse(matricesIguales(A, B))

    def test_mismos_valores_con_distinto_tipo(self):
        A = np.array([[1, 2], [3, 4]], dtype=int)
        B = np.array([[1, 2], [3, 4]], dtype=float)
        self.assertTrue(matricesIguales(A, B))

    def test_valores_claramente_distintos(self):
        A = np.eye(2)
        B = np.array([[1, 0], [0, 2]])
        self.assertFalse(matricesIguales(A, B))

    def test_diferencia_de_redondeo_se_considera_igual(self):
        A = np.eye(2)
        B = A.copy()
        B[0, 0] += np.finfo("float64").eps
        self.assertTrue(matricesIguales(A, B))

    def test_diferencia_mayor_a_la_tolerancia_no_es_igual(self):
        A = np.eye(2)
        B = A.copy()
        B[0, 0] += 1e-4
        self.assertFalse(matricesIguales(A, B))


if __name__ == "__main__":
    unittest.main(verbosity=2)
