import numpy as np
import draft_parser
from parsers.geof_parser import parse_geof_file
from core.face import Face
from core.boundary import Boundary
from core.cell import Cell

geof_file_path = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/squares_test.geof"
geof_file_path = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/triangles_test.geof"
# ----------------------------------------------------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------------------------------------------------
(
    vertices_matrix,
    cells_vertices_connectivity_matrix,
    faces_vertices_connectivity_matrix,
    cells_faces_connectivity_matrix,
    nsets,
) = parse_geof_file(geof_file_path)
# ----------------------------------------------------------------------------------------------------------------------
# basis
# ----------------------------------------------------------------------------------------------------------------------
cell_basis = Basis(k, d)
face_basis = Basis(k, d - 1)
# ----------------------------------------------------------------------------------------------------------------------
# boudary conditions
# ----------------------------------------------------------------------------------------------------------------------
dirichlet_boundary_conditions = {
    "BOTTOM": [lambda x: 0.0, lambda y: 0.0],
    "TOP": [lambda x: 1.1 * x, None],
    "LEFT": [None, None],
    "RIGHT": [None, None],
}
neumann_boundary_conditions = {
    "BOTTOM": [None, None],
    "TOP": [None, None],
    "LEFT": [None, None],
    "RIGHT": [None, None],
}

boundaries = []
for boundary_name in nsets.keys():
    nset_face_connectivity_matrix = nsets[boundary_name]
    boundary = Boundary(
        boundary_name,
        nset_face_connectivity_matrix,
        dirichlet_boundary_conditions[boundary_name],
        neumann_boundary_conditions[boundary_name],
    )
    boundaries.append(boundary)
# ----------------------------------------------------------------------------------------------------------------------
# faces
# ----------------------------------------------------------------------------------------------------------------------
faces = []
for face_index, face_vertices_connectivity_matrix in enumerate(faces_vertices_connectivity_matrix):
    face_vertices = []
    for vertex_index in face_vertices_connectivity_matrix:
        face_vertices.append(vertices_matrix[vertex_index])
    face_vertices_matrix = np.array(face_vertices)
    face = Face(face_vertices_matrix, )
    for boundary in boundaries:
        if face_index in boundary.faces_index:
            # APPLY BOUNDARY CONSTRAINT




    # for nset_face_connectivity_matrix in nsets.values():
    #     if face_index in nset_face_connectivity_matrix:

    # name: str,
    # faces_index: List[int],
    # imposed_dirichlet: Callable,
    # imposed_neumann: Callable,
    # print("------")
    # print(face_vertices)
