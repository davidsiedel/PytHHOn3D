import argparse
import numpy as np
from typing import List
from numpy import ndarray as Mat

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


def build(
    mesh_file: str,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    behavior: Behavior,
    boundary_conditions: dict,
):
    """
    ====================================================================================================================
    Description :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Exemple :
    ====================================================================================================================
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Checking polynomial order consistency
    # ------------------------------------------------------------------------------------------------------------------
    is_polynomial_order_consistent(face_polynomial_order, cell_polynomial_order)
    # ------------------------------------------------------------------------------------------------------------------
    # Reading the mesh file, and extracting conectivity matrices
    # ------------------------------------------------------------------------------------------------------------------
    (
        problem_dimension,
        vertices,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        cells_faces_connectivity_matrix,
        cells_connectivity_matrix,
        nsets,
    ) = parse_mesh(mesh_file)
    # ------------------------------------------------------------------------------------------------------------------
    # Writing the vertices matrix as a numpy object
    # ------------------------------------------------------------------------------------------------------------------
    vertices = np.array(vertices)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing polynomial bases
    # ------------------------------------------------------------------------------------------------------------------
    face_basis = ScaledMonomial(face_polynomial_order, problem_dimension - 1)
    cell_basis = ScaledMonomial(cell_polynomial_order, problem_dimension)
    integration_order = 2 * max(face_polynomial_order, cell_polynomial_order)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Face objects
    # ------------------------------------------------------------------------------------------------------------------
    faces = []
    for i, face_vertices_connectivity_matrix in enumerate(faces_vertices_connectivity_matrix):
        for boundary_name, nset in zip(nsets, nsets.values()):
            face_vertices = vertices[face_vertices_connectivity_matrix]
            # ----------------------------------------------------------------------------------------------------------
            # Checking whether the face belongs to a boundary or not : by scanning the nsets and connecting with the
            # boundary conditions given as argument of the function
            # ----------------------------------------------------------------------------------------------------------
            if i in nset:
                displacement = boundary_conditions[boundary_name][0]
                pressure = boundary_conditions[boundary_name][1]
                face = Face(face_vertices, integration_order, displacement=displacement, pressure=pressure)
            else:
                face = Face(face_vertices, integration_order)
        faces.append(face)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Cell objects
    # ------------------------------------------------------------------------------------------------------------------
    cells = []
    for cell_vertices_connectivity_matrix, cell_connectivity_matrix in zip(
        cells_vertices_connectivity_matrix, cells_connectivity_matrix
    ):
        cell_vertices = vertices[cell_vertices_connectivity_matrix]
        cell = Cell(cell_vertices, cell_connectivity_matrix, integration_order)
        cells.append(cell)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Elements objects
    # ------------------------------------------------------------------------------------------------------------------
    elements = []
    for i, cell in enumerate(cells):
        local_faces = [faces[j] for j in cells_faces_connectivity_matrix[i]]
        op = get_operator(
            operator_type, cell, local_faces, cell_basis, face_basis, problem_dimension, problem_dimension
        )
        # --------------------------------------------------------------------------------------------------------------
        local_cell_mass_matrix = op.local_cell_mass_matrix
        # local_face_mass_matrix = op.local_face_mass_matrix
        local_identity_operator = op.local_identity_operator
        local_reconstructed_gradient_operators = op.local_reconstructed_gradient_operators
        local_stabilization_matrix = op.local_stabilization_matrix
        local_load_vectors = op.local_load_vectors
        local_pressure_vectors = op.local_pressure_vectors
        # --------------------------------------------------------------------------------------------------------------
        del op
        vertices = cells_vertices_connectivity_matrix[i]
        quadrature_points: cell.quadrature_nodes
        del cell


def is_polynomial_order_consistent(face_polynomial_order: int, cell_polynomial_order: int):
    """
    ====================================================================================================================
    Description :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Exemple :
    ====================================================================================================================
    """
    if not face_polynomial_order in [cell_polynomial_order - 1, cell_polynomial_order, cell_polynomial_order + 1]:
        raise ValueError(
            "The face polynomial order must be the same order as the cell polynomial order or one order lower or greater"
        )


def get_operator(
    operator_type: str,
    cell: Cell,
    faces: List[Face],
    cell_basis: Basis,
    face_basis: Basis,
    problem_dimension: int,
    field_dimension: int,
):
    """
        ================================================================================================================
        Description :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        
        ================================================================================================================
        Exemple :
        ================================================================================================================
        """
    if operator_type == "HDG":
        op = HDG(cell, faces, cell_basis, face_basis, problem_dimension, field_dimension)
    else:
        raise NameError("The specified operator does not exist")
    return op


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Solve a mechanical system using the HHO method")
#     parser.add_argument("-msh", "--mesh_file", help="shows output")
#     parser.add_argument("-kf", "--face_polynomial_order", help="shows output")
#     parser.add_argument("-kc", "--cell_polynomial_order", help="shows output")
#     parser.add_argument("-op", "--operator_type", help="shows output")
#     args = parser.parse_args()
#     # ------------------------------------------------------------------------------------------------------------------
#     mesh_file = args.mesh_file
#     face_polynomial_order = int(args.face_polynomial_order)
#     cell_polynomial_order = int(args.cell_polynomial_order)
#     operator_type = args.operator_type
#     # ------------------------------------------------------------------------------------------------------------------
#     main(mesh_file, face_polynomial_order, cell_polynomial_order, operator_type)
