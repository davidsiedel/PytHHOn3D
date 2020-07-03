from tests import context

from core.face import Face
from core.cell import Cell
from core.element import Element
from bases.monomial import ScaledMonomial
import numpy as np
import pytest
from scipy import integrate

# ======================================================================================================================
# SEGMENT element
# ======================================================================================================================


def test_element():
    polynomial_order = 1
    cell_vertices = np.array([[1.0], [2.0]])
    faces_vertices = [np.array([[1.0]]), np.array([[2.0]])]
    connectivity_matrix = np.array([[0], [1]])
    integration_order = 2 * polynomial_order
    faces = [Face(face_vertices, integration_order) for face_vertices in faces_vertices]
    cell = Cell(cell_vertices, connectivity_matrix, integration_order)
    face_basis = ScaledMonomial(polynomial_order, 0)
    cell_basis = ScaledMonomial(polynomial_order, 1)
    direction = 0
    derivative_direction = 0
    element = Element(cell, faces, cell_basis, face_basis, direction, derivative_direction)
