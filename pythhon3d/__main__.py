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
from behaviors.behavior import Behavior
from behaviors.laplacian import Laplacian

from pythhon3d import build, solve

d = 2

if d == 1:
    k_f = 1
    k_c = 2
    dimension = 1
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c1d2.geof"
    # mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/test1D.geof"
    face_polynomial_order = k_f
    cell_polynomial_order = k_c
    operator_type = "HDG"
    pressure = [lambda x: 0.0]
    displacement = [lambda x: 0.0]
    displacement_tip = [lambda x: 1.0]
    load = [lambda x: np.sin(2.0 * np.pi * x[0])]
    load = [lambda x: np.sin(1.0 * np.pi * x[0])]
    # load = [lambda x: 0.0]
    boundary_conditions = {"RIGHT": (displacement, pressure), "LEFT": (displacement, pressure)}
    # boundary_conditions = {"RIGHT": (displacement_tip, pressure), "LEFT": (displacement, pressure)}
if d == 2:
    k_f = 1
    k_c = 1
    dimension = 2
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c2d3.geof"
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/triangles_test.geof"
    face_polynomial_order = k_f
    cell_polynomial_order = k_c
    operator_type = "HDG"
    pressure = [lambda x: 0.0, lambda x: 0.0]
    displacement = [lambda x: 0.0, lambda x: 0.0]
    load = [lambda x: np.sin(2.0 * np.pi * x[0] * x[1]), lambda x: np.sin(2.0 * np.pi * x[0] * x[1])]
    boundary_conditions = {
        "BOTTOM": (displacement, pressure),
        "TOP": (displacement, pressure),
        "RIGHT": (displacement, pressure),
        "LEFT": (displacement, pressure),
    }
if d == 3:
    k_f = 1
    k_c = 1
    dimension = 3
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c3d4.geof"
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
    ) = build(mesh_file, face_polynomial_order, cell_polynomial_order, operator_type)
    # ------------------------------------------------------------------------------------------------------------------
    tangent_matrices = [np.eye(d ** 2) for i in range(len(cells))]
    # ------------------------------------------------------------------------------------------------------------------
    vertices, vertices_sols, global_solution = solve(
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
        boundary_conditions,
        load,
    )
    import matplotlib.pyplot as plt

    print(global_solution.shape)
    plt.plot(vertices, global_solution[:100], label="HHO faces")
    plt.plot(vertices, vertices_sols, label="HHO cell")
    plt.plot(vertices, [((-2.0 * np.pi) ** 2) * load[0](x) for x in vertices], label="analytical")
    plt.legend()
    plt.show()
