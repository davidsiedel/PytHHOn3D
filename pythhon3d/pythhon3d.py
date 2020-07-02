from parsers import geof_parser as parser


def solve(mesh_file_path):
    (
        problem_dimension,
        vertices_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        cells_faces_connectivity_matrix,
        nsets,
    ) = parser.parse_geof_file(mesh_file_path)

    return
