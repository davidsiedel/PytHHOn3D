import argparse
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
rc("text", usetex=True)

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
# stabilization_parameter = 1000000.0  # K2
mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/mesh2D25.geof"
mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/2D/25.geof"
# mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/triangles_test.geof"
operator_type = "HDG"
operator_type = "HHO"
#
pressure_left = [None]
pressure_right = [None]
pressure_top = [None]
pressure_bottom = [None]
#
displacement_left = [lambda x: 0.0]
displacement_right = [lambda x: 0.0]
displacement_top = [None]
displacement_bottom = [None]
#
load = [lambda x: np.sin(2.0 * np.pi * x[0])]
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
    nsets_faces,
    cell_basis_l,
    cell_basis_k,
    face_basis_k,
    unknown,
) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
# ------------------------------------------------------------------------------------------------------------------
d = unknown.problem_dimension
tangent_matrices = [np.eye(2) for i in range(len(cells))]
# tangent_matrices = [np.array([[1.0, 0.0], [0.0, 1.0]]) for i in range(len(cells))]
# ------------------------------------------------------------------------------------------------------------------
(
    (vertices, unknowns_at_vertices),
    (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
    (vertices, f_unknowns_at_vertices),
    (x_cell_list, x_faces_list),
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
    cell_basis_l,
    cell_basis_k,
    face_basis_k,
    unknown,
    tangent_matrices,
    stabilization_parameter,
    boundary_conditions,
    load,
)
# ------------------------------------------------------------------------------------------------------------------

col_levels = 10
# vmaxx = 0.032
# vminn = -0.032
x, y = quadrature_points.T
# levels = np.linspace(vminn, vmaxx, col_levels)
error = np.array(
    (unknowns_at_quadrature_points[0] - np.array([-(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * X)) for X in x]))
)
error = error / 0.032
plt.tricontourf(x, y, error, cmap=cm.binary)
plt.xlabel("map of the domain $\Omega$")
cbar = plt.colorbar()
cbar.set_label("HHO error", rotation=270, labelpad=15.0)
plt.show()
