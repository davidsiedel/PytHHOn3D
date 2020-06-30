from pythhon3d.core.quadrature import Quadrature
import numpy as np
import pytest

# ----------------------------------------------------------------------------------------
# DIMENSION 0
# ----------------------------------------------------------------------------------------


def test_unite_point_quadrature():
    q = Quadrature("DUNAVANT")
    quadrature_nodes, quadrature_weights = q.get_unite_point_quadrature()
    expected_nodes, expected_weights = np.array([[]]), np.array([[1.0]])
    assert (quadrature_nodes == expected_nodes).all()
    assert (quadrature_weights == expected_weights).all()


# ----------------------------------------------------------------------------------------
# DIMENSION 1
# ----------------------------------------------------------------------------------------
# nodes = np.array([[0.0], [1.0]])
# volume = 1.
# ----------------------------------------------------------------------------------------

nodes_p1 = np.array([[0.5000000000000000]])
weights_p1 = np.array([[1.0000000000000000]])
expected_p1 = [nodes_p1, weights_p1]

nodes_p2 = np.array([[0.2113248654051871], [0.7886751345948129]])
weights_2 = np.array([[0.5000000000000000], [0.5000000000000000]])
expected_p2 = [nodes_p2, weights_2]

nodes_p3 = np.array([[0.1127016653792583], [0.5000000000000000], [0.8872983346207417]])
weights_p3 = np.array([[0.2777777777777778], [0.4444444444444444], [0.2777777777777778]])
expected_p3 = [nodes_p3, weights_p3]

test_data = [
    (np.array([[0.0], [1.0]]), 1.0, 1, expected_p1),
    (np.array([[0.0], [1.0]]), 1.0, 2, expected_p2),
    (np.array([[0.0], [1.0]]), 1.0, 3, expected_p2),
    (np.array([[0.0], [1.0]]), 1.0, 4, expected_p3),
    (np.array([[0.0], [1.0]]), 1.0, 5, expected_p3),
]


@pytest.mark.parametrize("nodes, volume, k, expected", test_data)
def test_unite_segment_quadrature(nodes, volume, k, expected):
    q = Quadrature("DUNAVANT")
    quadrature_nodes, quadrature_weights = q.get_unite_segment_quadrature(
        nodes, volume, k
    )
    assert (np.abs(quadrature_nodes - expected[0]) < np.full((k,), 1.0e-5)).all()
    assert (np.abs(quadrature_weights - expected[1]) < np.full((k,), 1.0e-5)).all()


polynom_1 = lambda x: 2.0 * x ** 1 + 3.0
integral_1 = 4.0

polynom_2 = lambda x: 3.0 * x ** 2 + 2.0 * x ** 1 + 7.0
integral_2 = 9.0

polynom_3 = lambda x: 5.0 * x ** 3 + 3.0 * x ** 2 + 2.0 * x + 2.0
integral_3 = 5.25

test_data = [
    (1, polynom_1, integral_1),
    (2, polynom_2, integral_2),
    (3, polynom_3, integral_3),
]


@pytest.mark.parametrize("k, polynom, expected", test_data)
def test_integral(k, polynom, expected):
    q = Quadrature("DUNAVANT")
    quadrature_nodes, quadrature_weights = q.get_unite_segment_quadrature(
        np.array([[0.0], [1.0]]), 1.0, k
    )
    # ------------------------------------------------------------------------------------
    # defining an arbitrary fucntion func and checking if the integral over the unite
    # segment is correct.
    # ------------------------------------------------------------------------------------
    volume = 1.0
    descrete_integral = np.sum(
        [
            weight / volume * polynom(node)
            for node, weight in zip(quadrature_nodes, quadrature_weights)
        ]
    )
    assert np.abs(descrete_integral - expected) < 1.0e-9
