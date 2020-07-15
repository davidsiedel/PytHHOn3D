import argparse
import numpy as np

from parsers.geof_parser import parse_geof_file as parse_mesh
from core.face import Face
from core.cell import Cell


def main(mesh_file, face_polynomial_order, cell_polynomial_order):
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
    # Reading the mesh file, and extracting conectivity matrices
    # ------------------------------------------------------------------------------------------------------------------
    (
        problem_dimension,
        vertices,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        cells_faces_connectivity_matrix,
        nsets,
    ) = parse_mesh(mesh_file)
    # ------------------------------------------------------------------------------------------------------------------
    # Writing the vertices matrix as a numpy object
    # ------------------------------------------------------------------------------------------------------------------
    vertices = np.array(vertices)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Face objects
    # ------------------------------------------------------------------------------------------------------------------
    faces = []
    # ------------------------------------------------------------------------------------------------------------------
    for face_vertices_connectivity_matrix in faces_vertices_connectivity_matrix:
        face_vertices = vertices[face_vertices_connectivity_matrix]
        face = Face(face_vertices, polynomial_order)
        faces.append(face)

    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Cell objects
    # ------------------------------------------------------------------------------------------------------------------
    cells = []
    # ------------------------------------------------------------------------------------------------------------------
    for cell_vertices_connectivity_matrix, cell_faces_connectivity_matrix in zip(
        cells_vertices_connectivity_matrix, cells_faces_connectivity_matrix
    ):
        cell_vertices = vertices[cell_vertices_connectivity_matrix]
        cell = Cell(cell_vertices, cell_faces_connectivity_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a mechanical system using the HHO method")
    parser.add_argument("-msh", "--mesh_file", help="shows output")
    parser.add_argument("-kf", "--face_polynomial_order", help="shows output")
    parser.add_argument("-kc", "--cell_polynomial_order", help="shows output")
    args = parser.parse_args()
    main(args.mesh_file, args.face_polynomial_order, args.cell_polynomial_order)
