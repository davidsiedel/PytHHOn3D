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

d = 1

if d == 1:
    k_f = 1
    k_c = 1
    dimension = 1
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c1d2.geof"
    # mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/test1D.geof"
    face_polynomial_order = k_f
    cell_polynomial_order = k_c
    operator_type = "HDG"
    pressure = [lambda x: 0.0]
    displacement = [lambda x: 0.0]
    load = [lambda x: np.sin(x)]
    # pressure = [lambda x: np.array([0.0])]
    # displacement = [lambda x: np.array([0.0])]
    # load = lambda x: [np.array([np.sin(x)])]
    boundary_conditions = {"RIGHT": (displacement, pressure), "LEFT": (displacement, pressure)}
if d == 2:
    k_f = 1
    k_c = 1
    dimension = 2
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c2d3.geof"
    face_polynomial_order = k_f
    cell_polynomial_order = k_c
    operator_type = "HDG"
    pressure = [lambda x: 0.0, lambda x: 0.0]
    displacement = [lambda x: 0.0, lambda x: 0.0]
    load = [lambda x: np.sin(x[0] * x[1]), lambda x: np.sin(x[0] * x[1])]
    # pressure = [lambda x: np.array([0.0])]
    # displacement = [lambda x: np.array([0.0])]
    # load = lambda x: [np.array([np.sin(x)])]
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
        lambda x: np.sin(x[0] * x[1] * x[2]),
        lambda x: np.sin(x[0] * x[1] * x[2]),
        lambda x: np.sin(x[0] * x[1] * x[2]),
    ]
    # pressure = [lambda x: np.array([0.0])]
    # displacement = [lambda x: np.array([0.0])]
    # load = lambda x: [np.array([np.sin(x)])]
    boundary_conditions = {
        "BOTTOM": (displacement, pressure),
        "TOP": (displacement, pressure),
        "RIGHT": (displacement, pressure),
        "LEFT": (displacement, pressure),
        "FRONT": (displacement, pressure),
        "BACK": (displacement, pressure),
    }

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Solve a mechanical system using the HHO method")
    # parser.add_argument("-msh", "--mesh_file", help="shows output")
    # parser.add_argument("-kf", "--face_polynomial_order", help="shows output")
    # parser.add_argument("-kc", "--cell_polynomial_order", help="shows output")
    # parser.add_argument("-op", "--operator_type", help="shows output")
    # args = parser.parse_args()
    # # ------------------------------------------------------------------------------------------------------------------
    # mesh_file = args.mesh_file
    # face_polynomial_order = int(args.face_polynomial_order)
    # cell_polynomial_order = int(args.cell_polynomial_order)
    # operator_type = args.operator_type
    # # -----------------------------------------------------------------------------------------------------------------
    # build(mesh_file, face_polynomial_order, cell_polynomial_order, operator_type, behavior, boundary_conditions, load)
    # build(mesh_file, face_polynomial_order, cell_polynomial_order, operator_type)
    (
        # vertices,
        # elements,
        # cells_faces_connectivity_matrix,
        # cells_vertices_connectivity_matrix,
        # faces_vertices_connectivity_matrix,
        # nsets,
        # cell_basis,
        # face_basis,
        # number_of_cells,
        # number_of_faces,
        # unknown,
        vertices,
        elements,
        faces,
        cells_faces_connectivity_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        nsets,
        cell_basis,
        face_basis,
        unknown,
    ) = build(mesh_file, face_polynomial_order, cell_polynomial_order, operator_type)
    # ------------------------------------------------------------------------------------------------------------------
    tangent_matrices = [np.eye(d ** 2) for i in range(len(elements))]
    # print(tangent_matrices[0])
    # ------------------------------------------------------------------------------------------------------------------
    solve(
        # elements,
        # cells_faces_connectivity_matrix,
        # cells_vertices_connectivity_matrix,
        # faces_vertices_connectivity_matrix,
        vertices,
        elements,
        faces,
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

