from parsers import geof_parser as parser
from shapes.shape_type import ShapeType


def solve(mesh_file_path):
    (
        problem_dimension,
        vertices_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        cells_faces_connectivity_matrix,
        nsets,
    ) = parser.parse_geof_file(mesh_file_path)

    shapes = ShapeType(problem_dimension)
    shapes.face_shape_type
    shapes.cell_shape_type

    if problem_dimension == 1:
        face_shape = "POINT"
        cell_shape = "LINE"
    elif problem_dimension == 2:
        face_shape = "LINE"
        cell_shape = "PLANE"
    elif problem_dimension == 3:
        face_shape = "PLANE"
        cell_shape = "VOLUME"

    return
