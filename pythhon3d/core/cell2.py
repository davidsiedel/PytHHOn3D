import numpy as np
from numpy import ndarray as Mat
from typing import List

from shapes.domain import Domain

from shapes.segment import Segment
from shapes.triangle import Triangle
from shapes.polygon import Polygon
from shapes.tetrahedron import Tetrahedron
from shapes.polyhedron import Polyhedron


# from core.domain import Domain
# from core.quadrature import Quadrature
# from core.boundary import Boundary


class Cell(Domain):
    def __init__(self, shape: str, vertices: Mat, polynomial_order: int):



import numpy as np
from numpy import ndarray as Mat
from core.quadrature import Quadrature
from core.domain import Domain
from core.face import Face
from typing import List
from typing import Callable


class Cell(Domain):
    def __init__(
        self,
        cell_vertices_matrix: Mat,
        faces: List[Face],
        faces_vertices_matrix: List[Mat],
        internal_load: Callable,
        k: int,
    ):
        ""
        ""
        super().__init__(cell_vertices_matrix)

        # --------------------------------------------------------------------------------------------------------------
        # initializing data
        # --------------------------------------------------------------------------------------------------------------
        cell_volume = 0.0
        signs = []
        cell_quadrature_points, cell_quadrature_weights = [], []
        # --------------------------------------------------------------------------------------------------------------
        # for every face
        # --------------------------------------------------------------------------------------------------------------
        for face, face_vertices_matrix in zip(faces, faces_vertices_matrix):
            vector_to_face = self.get_vector_to_face(face)
            distance_to_face = vector_to_face[:, -1][0]
            if distance_to_face > 0.0:
                sign = 1
            else:
                sign = -1
            signs.append(sign)
            distance_to_face = np.abs(distance_to_face)

    def get_vector_to_face(self, face: Face) -> Mat:
        """
        """
        p = face.reference_frame_transformation_matrix
        cell_barycenter = self.barycenter_vector
        face_barycenter = face.barycenter_vector
        vector_to_face = (p @ (cell_barycenter - face_barycenter).T).T
        return vector_to_face

    def get_cell_partition(self, faces_vertices_matrix, cell_vertices_matrix) -> Mat:
        ""
        ""
        d = cell_vertices_matrix.shape[1]
        number_of_faces = faces_vertices_matrix.shape[0]
        # --------------------------------------------------------------------------------------------------------------
        # Reading the problem dimension
        # --------------------------------------------------------------------------------------------------------------
        if d == 1:
            simplicial_sub_cells = [cell_vertices_matrix]
        if d == 2:
            if number_of_faces > d + 1:
                simplicial_sub_cells = []
                for i in range(number_of_faces):
                    sub_cell_vertices_matrix = [
                        face_vertices_matrix_in_face_reference_frame[i - 1, :],
                        face_vertices_matrix_in_face_reference_frame[i, :],
                        face_barycenter_vector_in_face_reference_frame[0],
                    ]
                    simplicial_sub_faces.append(np.array(sub_face_vertices_matrix))
            # ----------------------------------------------------------------------------------------------------------
            # getting the number of vertices.
            # ----------------------------------------------------------------------------
            number_of_vertices = face_vertices_matrix_in_face_reference_frame.shape[0]
            # ----------------------------------------------------------------------------
            # If the face is no simplex (i.e. of the number of vertices is greater than
            # (d-1)+1 = d), proceed to its segmentation in simplices.
            # ----------------------------------------------------------------------------
            if number_of_vertices > d + 1:
                simplicial_sub_faces = []
                for i in range(number_of_vertices):
                    sub_face_vertices_matrix = [
                        face_vertices_matrix_in_face_reference_frame[i - 1, :],
                        face_vertices_matrix_in_face_reference_frame[i, :],
                        face_barycenter_vector_in_face_reference_frame[0],
                    ]
                    simplicial_sub_faces.append(np.array(sub_face_vertices_matrix))
            else:
                simplicial_sub_faces = [face_vertices_matrix_in_face_reference_frame]
        return simplicial_sub_faces
        return

    def get_sub_cell_volume(self, face: Face, distance_to_face: float) -> Mat:
        ""
        ""
        d = face.barycenter_vector.shape[1]
        if d == 3:
            sub_cell_volume = (1.0 / 3.0) * face.volume * distance_to_face
        if d == 2:
            sub_cell_volume = (1.0 / 2.0) * face.volume * distance_to_face
        if d == 1:
            sub_cell_volume = (1.0 / 1.0) * face.volume * distance_to_face
        return sub_cell_volume
