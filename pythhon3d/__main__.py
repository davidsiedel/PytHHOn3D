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

from pythhon3d import build

mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c1d2.geof"
face_polynomial_order = 1
cell_polynomial_order = 1
operator_type = "HDG"
behavior = Laplacian(1)
pressure = [lambda x: 0.0]
displacement = [lambda x: 0.0]
load = [lambda x: np.sin(x)]
# pressure = [lambda x: np.array([0.0])]
# displacement = [lambda x: np.array([0.0])]
# load = lambda x: [np.array([np.sin(x)])]
boundary_conditions = {"RIGHT": (displacement, pressure), "LEFT": (displacement, pressure)}

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
    # # ------------------------------------------------------------------------------------------------------------------
    build(mesh_file, face_polynomial_order, cell_polynomial_order, operator_type, behavior, boundary_conditions)
