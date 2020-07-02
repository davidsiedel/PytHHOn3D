from tests import context

from core.face2 import Face
import numpy as np
import pytest
from scipy import integrate

# shape: str, vertices: Mat, polynomial_order: int
test_data = []
test_data.append((np.array([[0.1]]), np.array([0.1])))  # POINT
test_data.append((np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.5, 0.0])))  # SEGMENT
test_data.append(
    (np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),)  # TRIANGLE
)
test_data.append(
    (
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([1.0 / 2.0, 1.0 / 2.0, 0.0]),
    )  # POLYGON
)


@pytest.mark.parametrize("vertices, expected", test_data)
def test_face_centroid(vertices, expected):
    polynomial_order = 1
    face = Face(vertices, polynomial_order)
    assert (
        np.abs(face.centroid - expected) < np.full(expected.shape, 1.0e-9)
    ).all() and face.centroid.shape == expected.shape


test_data = []
test_data.append((np.array([[0.1]]), 1.0))  # POINT
test_data.append((np.array([[0.0, 0.0], [1.0, 0.0]]), 1.0))  # SEGMENT
test_data.append((np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 1.0 / 2.0,))  # TRIANGLE
test_data.append(
    (np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), 1.0)  # POLYGON
)


@pytest.mark.parametrize("vertices, expected", test_data)
def test_face_volume(vertices, expected):
    polynomial_order = 1
    face = Face(vertices, polynomial_order)
    assert np.abs(face.volume - expected) < 1.0e-9


# ======================================================================================================================

polynomial_orders = [1, 2, 3, 4, 5]

functions_segment = [
    (lambda x: 2.0 * x ** 1 + 3.0),
    (lambda x: 1.0 * x ** 2 + 6.0 * x ** 1 - 2.0),
    (lambda x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x ** 1 - 1.0),
    (lambda x: 7.0 * x ** 4 - 4.0 * x ** 3 + 5.0 * x ** 2 - 1.0 * x ** 1 + 9.0),
    (lambda x: 2.0 * x ** 5 - 1.0 * x ** 4 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x ** 1 - 6.0),
]

test_data = []
height = 1.3
for k in polynomial_orders:
    test_data.append((k, functions_segment[k - 1](height)))


@pytest.mark.parametrize("k, expected", test_data)
def test_point_face_quadrature(k, expected):
    vertices = np.array([[1.3]])
    face = Face(vertices, k)
    numerical_integral = np.sum(
        [
            quadrature_weight * functions_segment[k - 1](quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(face.quadrature_nodes, face.quadrature_weights)
        ]
    )
    assert np.abs(numerical_integral - expected) < 1.0e-9


# ======================================================================================================================

polynomial_orders = [1, 2, 3, 4, 5]

functions_polygon = [
    (lambda y, x: 2.0 * x ** 1 + 3.0 * y ** 1 + 7.0),
    (lambda y, x: 1.0 * x ** 2 + 6.0 * x + 3.0 * y ** 2 - 2.0 * y + 2.0 * x * y - 2.0),
    (lambda y, x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x - 1.0 + 4.0 * y ** 3),
    (lambda y, x: 7.0 * x ** 4 - 4.0 * x ** 3 + 5.0 * x ** 2 - 1.0 * x + 9.0 + 5.0 * y ** 4),
    (lambda y, x: 2.0 * x ** 5 - 1.0 * x ** 4 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x - 6.0 + 6.0 * y ** 5),
]

test_data = []
height = 1.3
for k in polynomial_orders:
    temp_fun = lambda x: functions_polygon[k - 1](height, x)
    test_data.append((k, integrate.quad(temp_fun, 0.0, 1.0)[0]))
    # test_data.append((k, integrate.quad(functions_segment[k - 1], 0.0, 1.0)[0]))
    # test_data.append((k, integrate.dblquad(functions_polygon[k - 1], 0.0, 1.0, lambda x: 0.0, lambda x: 1)[0]))


@pytest.mark.parametrize("k, expected", test_data)
def test_polygon_quadrature(k, expected):
    vertices = np.array([[1.0, 1.3], [0.0, 1.3]])
    face = Face(vertices, k)
    # polygon = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # t = polygon(polygon, k)
    numerical_integral = np.sum(
        [
            quadrature_weight * functions_polygon[k - 1](quadrature_point[1], quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(face.quadrature_nodes, face.quadrature_weights)
        ]
    )
    assert np.abs(numerical_integral - expected) < 1.0e-9


# ======================================================================================================================


polynomial_orders = [1, 2, 3, 4, 5]
functions_polyhedron = [
    (lambda z, y, x: 2.0 * x ** 1 + 3.0 * y ** 1 + 7.0 + 2.0 * z),
    (lambda z, y, x: 1.0 * x ** 2 + 6.0 * x + 3.0 * y ** 2 - 2.0 * y + 2.0 * x * y - 2.0 + 4.0 * z * y - 2.5 * z * x),
    (lambda z, y, x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x - 1.0 + 4.0 * y ** 3 + 4.0 * z ** 2 * y - 2.5 * z * x ** 2),
    (
        lambda z, y, x: 7.0 * x ** 4
        - 4.0 * x ** 3
        + 5.0 * x ** 2
        - 1.0 * x
        + 9.0
        + 5.0 * y ** 4
        + 4.0 * z * y
        - 2.5 * z ** 3 * x
    ),
    (
        lambda z, y, x: 2.0 * x ** 5
        - 1.0 * x ** 4
        + 1.0 * x ** 3
        - 3.0 * x ** 2
        + 1.0 * x
        - 6.0
        + 6.0 * y ** 5
        + 4.0 * z ** 4 * y
        - 2.5 * z * x
    ),
]

test_data = []
height = 1.3
for k in polynomial_orders:
    temp_fun = lambda y, x: functions_polyhedron[k - 1](height, y, x)
    test_data.append((k, integrate.dblquad(temp_fun, 0.0, 1.0, lambda x: 0.0, lambda x: 1)[0]))


@pytest.mark.parametrize("k, expected", test_data)
def test_polygonal_face_quadrature(k, expected):
    vertices = np.array(
        [[0.0, 0.0, height], [1.0, 0.0, height], [1.0, 0.5, height], [1.0, 1.0, height], [0.0, 1.0, height]]
    )
    face = Face(vertices, k)
    numerical_integral = np.sum(
        [
            quadrature_weight
            * functions_polyhedron[k - 1](quadrature_point[2], quadrature_point[1], quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(face.quadrature_nodes, face.quadrature_weights)
        ]
    )
    assert np.abs(numerical_integral - expected) < 1.0e-9
