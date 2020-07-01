from drafts import context
from shapes.polyhedron import Polyhedron
import numpy as np

polyhedron = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 1.0],
    ]
)
connectivity_matrix = [
    np.array([0, 1, 2, 3]),
    np.array([1, 2, 6, 5]),
    np.array([0, 1, 5, 4]),
    np.array([2, 3, 7, 6]),
    np.array([0, 3, 7, 4]),
    np.array([4, 8, 7]),
    np.array([4, 8, 5]),
    np.array([5, 8, 6]),
    np.array([6, 8, 7]),
]
polynomial_order = 1
p = Polyhedron(polyhedron, connectivity_matrix, polynomial_order)
