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
face_polynomial_order = 2
cell_polynomial_order = 2
field_dimension = 1
stabilization_parameter = 10.0  # K2
# stabilization_parameter = 1000000.0  # K2
mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/mesh2D25.geof"
# mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/triangles_test.geof"
operator_type = "HDG"
#
pressure_left = [lambda x: 0.0]
pressure_right = [lambda x: 0.0]
pressure_top = [lambda x: 0.0]
pressure_bottom = [lambda x: 0.0]
#
displacement_left = [None]
displacement_right = [None]
displacement_top = [lambda x: 1.0]
displacement_bottom = [lambda x: 0.0]
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
    flags,
    cell_basis,
    face_basis,
    unknown,
) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
# ------------------------------------------------------------------------------------------------------------------
d = unknown.problem_dimension
tangent_matrices = [np.eye(2) for i in range(len(cells))]
# tangent_matrices = [np.array([[1.0, 0.0], [0.0, 1.0]]) for i in range(len(cells))]
# ------------------------------------------------------------------------------------------------------------------
(
    (vertices, unknowns_at_vertices),
    (quadrature_points, unknowns_at_quadrature_points),
    (f_vertices, f_unknowns_at_vertices),
) = solve(
    vertices,
    faces,
    cells,
    operators,
    cells_faces_connectivity_matrix,
    cells_vertices_connectivity_matrix,
    faces_vertices_connectivity_matrix,
    nsets,
    flags,
    cell_basis,
    face_basis,
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

    x = np.linspace(0.0, 1.0, 100)
    y = np.linspace(0.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = 6.0 * Y - 5.0 * Y ** 2

    import matplotlib.cm as cm
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    col_levels = 10
    # CS = plt.contourf(X, Y, Z, 5, vmin=-0.2, vmax=2.0, cmap=cm.viridis)
    # CS = plt.contourf(X, Y, Z, 1000, vmin=-0.0, vmax=1.8066059951119915, cmap=cm.viridis)
    CS = plt.contourf(X, Y, Z, col_levels, vmin=-0.0, vmax=2.0, cmap=cm.viridis)
    # plt.title("Simplest default with labels")
    m = plt.cm.ScalarMappable(cmap=cm.viridis)
    m.set_array(Z)
    # m.set_clim(-0.2, 2.0)
    # m.set_clim(-0.0, 1.8066059951119915)
    m.set_clim(-0.0, 2.0)
    # plt.colorbar(m, boundaries=np.linspace(-0.2, 2.0, 1000))
    # plt.colorbar(m, boundaries=np.linspace(0.0, 1.8066059951119915, 1000))
    plt.colorbar(m, boundaries=np.linspace(0.0, 2.0, col_levels))
    plt.show()

    # fig, ax = plt.subplots()

    # bounds = np.linspace(0, 2.01, 12)
    # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=250)
    # pcm = ax.pcolormesh(X, Y, Z, norm=norm, cmap="viridis")
    # fig.colorbar(pcm, ax=ax, extend="neither", orientation="vertical", vmin=0, vmax=2)
    # # plt.pcolor(X, Y, Z, norm=norm)
    # # plt.colorbar()
    # # plt.colorbar()
    # plt.show()
    # -----------
    # x, y = f_vertices.T
    # plt.tricontourf(x, y, f_unknowns_at_vertices[0], levels=1000)
    # plt.colorbar()
    # plt.show()
    #
    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # for ax in axes.flat:
    #     im = plt.tricontourf(x, y, f_unknowns_at_vertices[0], levels=1000)

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # plt.show()
    #
    x, y = vertices.T
    print("MAXXXX : {}".format(max(unknowns_at_vertices[0])))
    print("MINNNN : {}".format(min(unknowns_at_vertices[0])))
    levels = np.linspace(0.0, 2.00, col_levels)
    plt.tricontourf(x, y, unknowns_at_vertices[0], levels=levels)
    plt.colorbar()
    plt.show()

    # import matplotlib.cm as cm
    # import matplotlib.mlab as mlab
    # import matplotlib.pyplot as plt

    # CS = plt.contourf(x, y, Z, unknowns_at_vertices[0], vmin=0.0, vmax=2.0, cmap=cm.viridis)
    # plt.title("Simplest default with labels")
    # m = plt.cm.ScalarMappable(cmap=cm.viridis)
    # m.set_array(Z)
    # m.set_clim(0.0, 2.0)
    # plt.colorbar(m, boundaries=np.linspace(0, 2, 12))
    # plt.show()

    # bounds = np.linspace(-1, 1, 10)
    # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # pcm = ax[0].pcolormesh(X, Y, Z, norm=norm, cmap="RdBu_r")
    # fig.colorbar(pcm, ax=ax[0], extend="both", orientation="vertical")


# z = unknowns_at_vertices[0]
# scaled_z = (z - z.min()) / z.ptp()
# colors = plt.cm.coolwarm(scaled_z)
# plt.scatter(vertices[:, 0], vertices[:, 1], c=colors, s=100, linewidths=4)
# # for pX, pY, val in zip(vertices[:, 0], vertices[:, 1], unknowns_at_vertices[0]):
# #     plt.scatter(pX, pY, val)

# plt.colorbar()
# plt.show()
# #
# import numpy as np
# import matplotlib.pyplot as plt

# x = np.linspace(0.0, 1.0, 100)
# y = np.linspace(0.0, 1.0, 100)
# X, Y = np.meshgrid(x, y)

# # Z = (1 - X / 2 + X ** 5 + Y ** 3) * np.exp(-(X ** 2) - Y ** 2)  # calcul du tableau des valeurs de Z
# Z = np.sin(2.0 * np.pi / 0.9 * X) * np.sin(2.0 * np.pi / 0.3 * Y)
# Z = Y * (1.0 - Y) * (X ** 3)

# plt.pcolor(X, Y, Z)
# plt.colorbar()

# # plt.show()
