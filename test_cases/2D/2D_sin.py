import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

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
face_polynomial_order = 0
cell_polynomial_order = 1
field_dimension = 1
stabilization_parameter = 1.0  # K2
# stabilization_parameter = 1000000.0  # K2
mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/mesh2D25.geof"
mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/2D/15.geof"
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
# displacement_top = [lambda x: 0.0]
# displacement_bottom = [lambda x: 0.0]
#

lam = 10.0
load = [
    lambda x: (0.2 * np.sin(2.0 * np.pi * x[1])) * (np.cos(2.0 * np.pi * x[0]) - 1.0)
    + (np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])) / (5.0 + 5.0 * lam),
]
#
load = [
    lambda x: 6.0 * x[0] * x[1] * (1.0 - x[1]) - 2.0 * x[0] ** 3,
]
#
load = [
    lambda x: -0.0,
]
#
load = [
    lambda x: np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]),
]
#
load = [
    lambda x: -10.0,
]
#
load = [lambda x: np.sin(2.0 * np.pi * x[0])]
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
scatter = False
if scatter:
    marker_size = 50
    plt.scatter(f_vertices[:, 0], f_vertices[:, 1], marker_size, c=f_unknowns_at_vertices[0])
    # plt.scatter(vertices[:, 0], vertices[:, 1], marker_size, c=unknowns_at_vertices[0])
    cbar = plt.colorbar()
    plt.show()

    plt.scatter(vertices[:, 0], vertices[:, 1], marker_size, c=unknowns_at_vertices[0])
    # plt.scatter(vertices[:, 0], vertices[:, 1], marker_size, c=unknowns_at_vertices[0])
    cbar = plt.colorbar()
    plt.show()
else:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cbook as cbook

    # fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    x = np.linspace(0.0, 1.0, 100)
    y = np.linspace(0.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    # Z = 6.0 * Y - 5.0 * Y ** 2
    Z = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * X))

    import matplotlib.cm as cm
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    col_levels = 10
    vmaxx = 0.032
    vminn = -0.032
    CS = plt.contourf(X, Y, Z, col_levels, vmin=vminn, vmax=vmaxx, cmap=cm.binary)
    m = plt.cm.ScalarMappable(cmap=cm.binary)
    m.set_array(Z)
    m.set_clim(vminn, vmaxx)
    cbar = plt.colorbar(m, boundaries=np.linspace(vminn, vmaxx, col_levels))
    cbar.set_label("Analytical solution", rotation=270, labelpad=15.0)
    # cbar.ax.set_title("Analytical solution")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("map of the domain $\Omega$")
    plt.show()
    # x, y = vertices.T
    x, y = quadrature_points.T
    levels = np.linspace(vminn, vmaxx, col_levels)
    # plt.tricontourf(x, y, unknowns_at_vertices[0], levels=levels)
    # plt.tricontourf(x, y, unknowns_at_vertices[0])
    error = unknowns_at_quadrature_points[0] - np.array(
        [-(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * X)) for X in x]
    )
    # plt.tricontourf(x, y, unknowns_at_quadrature_points[0], cmap=cm.binary, levels=levels)
    plt.tricontourf(x, y, error, cmap=cm.binary, levels=levels)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("map of the domain $\Omega$")
    cbar = plt.colorbar()
    cbar.set_label("HHO quadratic solution", rotation=270, labelpad=15.0)
    # cbar.ax.set_title("HHO quadratic solution")
    plt.show()
