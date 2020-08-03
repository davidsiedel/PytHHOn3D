import argparse
import numpy as np

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

if d == 1:
    k_f = 1
    k_c = 1
    if k_c == 1:
        stabilization_parameter = 1.0e10  # K1
        stabilization_parameter = 100000.0  # K1
    if k_c == 2:
        stabilization_parameter = 300.0e1  # K2
        stabilization_parameter = 1.0e5  # K2
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c1d2.geof"
    # mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/test1D.geof"
    field_dimension = 1
    face_polynomial_order = k_f
    cell_polynomial_order = k_c
    operator_type = "HDG"
    pressure = [lambda x: 0.0]
    pressure_left = [lambda x: 0.0]
    pressure_right = [lambda x: 0.0]
    displacement = [lambda x: 0.0]
    # displacement = [None]
    displacement_tip_right = [lambda x: -1.0 / 12.0]
    # displacement_tip_right = [None]
    displacement_tip_left = [lambda x: 0.0]
    # displacement_tip_right = [None]
    # displacement_tip_left = [None]
    # load = [lambda x: np.sin(2.0 * np.pi * x[0])]
    load = [lambda x: x[0] ** 2 - x[0]]
    # load = [lambda x: 0.0]
    # load = [lambda x: np.sin(1.0 * np.pi * x[0])]
    # load = [lambda x: 0.0]
    # boundary_conditions = {"RIGHT": (displacement, pressure), "LEFT": (displacement, pressure)}
    # boundary_conditions = {"RIGHT": (displacement_tip_right, pressure), "LEFT": (displacement_tip_left, pressure)}
    boundary_conditions = {
        "RIGHT": (displacement_tip_right, pressure_right),
        "LEFT": (displacement_tip_left, pressure_left),
    }
    # boundary_conditions = {"RIGHT": (displacement, pressure), "LEFT": (displacement, pressure)}
if d == 2:
    k_f = 1
    k_c = 2
    if k_c == 1:
        stabilization_parameter = 1.0e1  # K1
    if k_c == 2:
        stabilization_parameter = 10.0e1  # K2
        stabilization_parameter = 1.0e10  # K2
        stabilization_parameter = 1.0  # K2
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c2d3.geof"
    # mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/triangles_test.geof"
    field_dimension = 2
    face_polynomial_order = k_f
    cell_polynomial_order = k_c
    operator_type = "HDG"
    pressure = [lambda x: 0.0, lambda x: 0.0]
    displacement = [lambda x: 0.0, lambda x: 0.0]
    displacement_bottom = [lambda x: 0.0, lambda x: 0.0]
    displacement_left = [None, None]
    displacement_right = [None, None]
    displacement_top = [lambda x: 0.5, None]
    load = [
        lambda x: np.sin(2.0 * np.pi / 0.9 * x[0]) * np.sin(2.0 * np.pi / 0.3 * x[1]),
        lambda x: np.sin(2.0 * np.pi / 0.9 * x[0]) * np.sin(2.0 * np.pi / 0.3 * x[1]),
        # lambda x: np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]),
    ]
    load = [lambda x: 0.0, lambda x: 0.0]
    # pressure = [lambda x: 0.0]
    # displacement = [lambda x: 0.0]
    # load = [lambda x: np.sin(2.0 * np.pi * x[0] * x[1])]
    boundary_conditions = {
        "BOTTOM": (displacement_bottom, pressure),
        "TOP": (displacement_top, pressure),
        "RIGHT": (displacement_right, pressure),
        "LEFT": (displacement_left, pressure),
    }
if d == 3:
    k_f = 1
    k_c = 1
    stabilization_parameter = 1.0e10  # K1
    stabilization_parameter = 300.0e1  # K2
    dimension = 3
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c3d4.geof"
    field_dimension = 3
    face_polynomial_order = 1
    cell_polynomial_order = 1
    operator_type = "HDG"
    pressure = [lambda x: 0.0, lambda x: 0.0, lambda x: 0.0]
    displacement = [lambda x: 0.0, lambda x: 0.0, lambda x: 0.0]
    load = [
        lambda x: np.sin(2.0 * np.pi * x[0] * x[1] * x[2]),
        lambda x: np.sin(2.0 * np.pi * x[0] * x[1] * x[2]),
        lambda x: np.sin(2.0 * np.pi * x[0] * x[1] * x[2]),
    ]
    boundary_conditions = {
        "BOTTOM": (displacement, pressure),
        "TOP": (displacement, pressure),
        "RIGHT": (displacement, pressure),
        "LEFT": (displacement, pressure),
        "FRONT": (displacement, pressure),
        "BACK": (displacement, pressure),
    }

if __name__ == "__main__":
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
    tangent_matrices = [np.eye(d ** 2) for i in range(len(cells))]
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
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp2d as intp

    print(vertices)
    print(unknowns_at_vertices)

    if d == 1:
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
    if d == 2:
        x, y = vertices.T
        plt.tricontourf(x, y, unknowns_at_vertices[0])
        plt.colorbar()
        plt.show()
        #
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.linspace(0.0, 0.9, 100)
        y = np.linspace(0.0, 0.3, 50)
        X, Y = np.meshgrid(x, y)

        # Z = (1 - X / 2 + X ** 5 + Y ** 3) * np.exp(-(X ** 2) - Y ** 2)  # calcul du tableau des valeurs de Z
        Z = np.sin(2.0 * np.pi / 0.9 * X) * np.sin(2.0 * np.pi / 0.3 * Y)

        plt.pcolor(X, Y, Z)
        plt.colorbar()

        plt.show()

        # x, y = vertices[:, 0], vertices[:, 1]
        # z = intp(x, y, unknowns_at_vertices[0], kind="linear")
        # X_v, Y_v = np.meshgrid(x, y)
        # Z_v = z(X_v, Y_v)
        # plt.pcolor(X_v, Y_v, Z_v)
        # plt.show()

