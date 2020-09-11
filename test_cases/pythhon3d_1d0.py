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

d = 1
face_polynomial_order = 1
cell_polynomial_order = 2
field_dimension = 1
stabilization_parameter = 1.0e5  # K2
mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c1d2.geof"
operator_type = "HDG"
#
pressure_left = [lambda x: 0.0]
pressure_right = [lambda x: 0.0]
#
displacement_left = [lambda x: 0.0]
displacement_right = [lambda x: -1.0 / 12.0]
#
load = [lambda x: x[0] ** 2 - x[0]]
#
boundary_conditions = {
    "RIGHT": (displacement_right, pressure_right),
    "LEFT": (displacement_left, pressure_left),
}
# ------------------------------------------------------------------------------------------------------------------
# vertices,
#         faces,
#         cells,
#         operators,
#         cells_faces_connectivity_matrix,
#         cells_vertices_connectivity_matrix,
#         faces_vertices_connectivity_matrix,
#         nsets,
#         nsets_faces,
#         cell_basis,
#         face_basis,
#         unknown,
(
    vertices,
    faces,
    cells,
    operators,
    cells_faces_connectivity_matrix,
    cells_vertices_connectivity_matrix,
    faces_vertices_connectivity_matrix,
    nsets,
    nsets_faces,
    cell_basis,
    face_basis,
    unknown,
) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
# ------------------------------------------------------------------------------------------------------------------
d = unknown.problem_dimension
tangent_matrices = [np.eye(d ** 2) for i in range(len(cells))]
# ------------------------------------------------------------------------------------------------------------------
# vertices: Mat,
# faces: List[Face],
# cells: List[Cell],
# operators: List[Operator],
# cells_faces_connectivity_matrix: Mat,
# cells_vertices_connectivity_matrix: Mat,
# faces_vertices_connectivity_matrix: Mat,
# nsets: dict,
# # flags: List[str],
# nsets_faces: dict,
# cell_basis: Basis,
# face_basis: Basis,
# unknown: Unknown,
# tangent_matrices: List[Mat],
# stabilization_parameter: float,
# boundary_conditions: dict,
# load: List[Callable],
# (vertices, unknowns_at_vertices),
#         (quadrature_points, unknowns_at_quadrature_points),
#         (vertices, f_unknowns_at_vertices),
(
    (vertices, unknowns_at_vertices),
    (quadrature_points, unknowns_at_quadrature_points),
    (vertices, f_unknowns_at_vertices),
) = solve(
    vertices,
    faces,
    cells,
    operators,
    cells_faces_connectivity_matrix,
    cells_vertices_connectivity_matrix,
    faces_vertices_connectivity_matrix,
    nsets,
    nsets_faces,
    cell_basis,
    face_basis,
    unknown,
    tangent_matrices,
    stabilization_parameter,
    boundary_conditions,
    load,
)
# ------------------------------------------------------------------------------------------------------------------
x_axis_v = [vertices[i] for i in range(vertices.shape[0])]
y_axis_v = [unknowns_at_vertices[0][i] for i in range(unknowns_at_vertices[0].shape[0])]
x_axis_q = [quadrature_points[i] for i in range(quadrature_points.shape[0])]
y_axis_q = [unknowns_at_quadrature_points[0][i] for i in range(unknowns_at_quadrature_points[0].shape[0])]
plt.plot(x_axis_v, y_axis_v, label="HHO vertices")
# plt.plot(vertices, xf, label="HHO faces")
plt.plot(x_axis_q, y_axis_q, label="HHO quad")
# plt.plot(x_axis_v, [-(1.0 / (2.0 * np.pi) ** 2) * load[0](x) for x in x_axis_v], label="analytical")
# for segment in [(x_axis_v[i-1], x_axis_v[i]) for i in range(1, len(x_axis_v))]:
plt.plot(x_axis_v, [((x ** 4) / 12.0 - (x ** 3) / 6.0) for x in x_axis_v], label="analytical")
# plt.plot(x_axis_v, [0.1 * x for x in x_axis_v], label="analytical")
plt.legend()
plt.show()
