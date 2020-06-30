from pythhon3d.core.face import Face
from pythhon3d.core.quadrature import Quadrature
from pythhon3d.core.boundary import Boundary
import numpy as np
import pytest


# class Boundary:
#     def __init__(
#         self,
#         name: str,
#         faces_index: List[int],
#         imposed_dirichlet: Callable,
#         imposed_neumann: Callable,
#     ):
#         """
#         ejbchbjs
#         """
#         self.name = name
#         self.faces_index = faces_index
#         self.imposed_dirichlet = imposed_dirichlet
#         self.imposed_neumann = imposed_neumann
#         return

b = Boundary("TEST", [0.0], lambda x: 0.0, lambda x: 0.0)
k = 1
test_data = []
# ----------------------------------------------------------------------------------------
# DIMENSION 1
# ----------------------------------------------------------------------------------------
vertices_matrix = np.array([[0.0]])
expeced_barycenter = np.array([[0.0]])
test_data.append((vertices_matrix, expeced_barycenter))
# ----------------------------------------------------------------------------------------
# DIMENSION 2
# ----------------------------------------------------------------------------------------
vertices_matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
expeced_barycenter = np.array([1.0 / 3.0, 1.0 / 3.0])
test_data.append((vertices_matrix, expeced_barycenter))
# ----------------------------------------------------------------------------------------
# DIMENSION 3
# ----------------------------------------------------------------------------------------
vertices_matrix = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
expeced_barycenter = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
test_data.append((vertices_matrix, expeced_barycenter))

# faces = [f1, f2, f3]
# for face in faces:
#     face_barycenter_vector = face.get_face_barycenter_vector(vertices)
#     face_vertices_rolled_matrix = face.get_face_vertices_rolled_matrix(vertices)
#     face_vertices_differences_matrix = face.get_face_vertices_differences_matrix(
#         vertices, face_vertices_rolled_matrix
#     )
#     vol = face.get_simplex_volume(face_vertices_differences_matrix)


@pytest.mark.parametrize("vertices_matrix, expected_barycenter_vector", test_data)
def test_face_barycenter_vector(vertices_matrix, expected_barycenter_vector):
    face = Face(vertices_matrix, k, b)
    face_barycenter_vector = face.get_face_barycenter_vector(vertices_matrix)
    assert (
        np.abs(face_barycenter_vector - expected_barycenter_vector)
        < np.full(expected_barycenter_vector.shape, 1.0e-5)
    ).all()
