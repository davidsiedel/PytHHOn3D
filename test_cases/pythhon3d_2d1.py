import argparse
import numpy as np
import matplotlib.pyplot as plt

from test_cases import context

from parsers.geof_parser import parse_geof_file as parse_mesh
from parsers.element_types import C_cf_ref
from bases.basis import Basis
from bases.monomial import ScaledMonomial
from core.face import Face
from core.cell import Cell
from core.operators.operator import Operator
from core.operators.hdg import HDG

from pythhon3d import build, solve

d = 2
face_polynomial_order = 1
cell_polynomial_order = 2
field_dimension = 1
stabilization_parameter = 1.0  # K2
mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/mesh2D25.geof"
operator_type = "HDG"
#
pressure_left = [lambda x: 0.0]
pressure_right = [lambda x: 0.0]
pressure_top = [lambda x: 0.0]
pressure_bottom = [lambda x: 0.0]
#
displacement_left = [lambda x: 0.0]
displacement_right = [lambda x: 0.0]
displacement_top = [lambda x: 0.0]
displacement_bottom = [lambda x: 0.0]
#
load = [
    lambda x: np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]),
]
#
# lam = 10.0
# load = [
#     lambda x: (0.2 * np.sin(2.0 * np.pi * x[1])) * (np.cos(2.0 * np.pi * x[0]) - 1.0)
#     + (np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])) / (5.0 + 5.0 * lam),
# ]
# #
# load = [
#     lambda x: 6.0 * x[0] * x[1] * (1.0 - x[1]) - 2.0 * x[0] ** 3,
# ]
# #
# load = [
#     lambda x: 1.0,
# ]
#
boundary_conditions = {
    "RIGHT": (displacement_right, pressure_right),
    "LEFT": (displacement_left, pressure_left),
    "TOP": (displacement_top, pressure_top),
    "BOTTOM": (displacement_bottom, pressure_bottom),
}
# ------------------------------------------------------------------------------------------------------------------
(
    vertices,
    faces,
    cells,
    operators,
    cells_faces_connectivity_matrix,
    cells_vertices_connectivity_matrix,
    faces_vertices_connectivity_matrix,
    nsets,
    cell_basis,
    face_basis,
    unknown,
) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
# ------------------------------------------------------------------------------------------------------------------
d = unknown.problem_dimension
# tangent_matrices = [np.eye(d ** 2) for i in range(len(cells))]
tangent_matrices = [np.eye(2) for i in range(len(cells))]
# ------------------------------------------------------------------------------------------------------------------
(vertices, unknowns_at_vertices), (quadrature_points, unknowns_at_quadrature_points) = solve(
    vertices,
    faces,
    cells,
    operators,
    cells_faces_connectivity_matrix,
    cells_vertices_connectivity_matrix,
    faces_vertices_connectivity_matrix,
    nsets,
    cell_basis,
    face_basis,
    unknown,
    tangent_matrices,
    stabilization_parameter,
    boundary_conditions,
    load,
)
# ------------------------------------------------------------------------------------------------------------------
x, y = vertices.T
plt.tricontourf(x, y, unknowns_at_vertices[0])
plt.colorbar()
plt.show()
#
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.0, 1.0, 100)
y = np.linspace(0.0, 1.0, 100)
X, Y = np.meshgrid(x, y)

# Z = (1 - X / 2 + X ** 5 + Y ** 3) * np.exp(-(X ** 2) - Y ** 2)  # calcul du tableau des valeurs de Z
Z = np.sin(2.0 * np.pi / 0.9 * X) * np.sin(2.0 * np.pi / 0.3 * Y)
Z = Y * (1.0 - Y) * (X ** 3)

plt.pcolor(X, Y, Z)
plt.colorbar()

plt.show()
