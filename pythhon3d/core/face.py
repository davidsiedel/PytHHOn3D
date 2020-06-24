import numpy as np
from numpy import ndarray as Mat
from typing import List
from core.domain import Domain
from core.quadrature import Quadrature
from core.boundary import Boundary


class Face(Domain):
    def __init__(self, face_vertices_matrix: Mat, k: int):
        """
        """
        super().__init__(face_vertices_matrix)

        # print("face_vertices_matrix : \n{}\n".format(face_vertices_matrix))

        # --------------------------------------------------------------------------------------------------------------
        # Computing the mapping from the cell reference frame into the face hyperplane
        # --------------------------------------------------------------------------------------------------------------

        self.reference_frame_transformation_matrix = self.get_face_reference_frame_transformation_matrix(
            face_vertices_matrix
        )
        # print("reference_frame_transformation_matrix : \n{}\n".format(self.reference_frame_transformation_matrix))

        # --------------------------------------------------------------------------------------------------------------
        # Working in face dimension : projecting vertices and barycenter into the face hyperplane
        # --------------------------------------------------------------------------------------------------------------

        face_vertices_matrix_in_face_reference_frame = self.get_points_in_face_reference_frame(face_vertices_matrix)
        face_barycenter_vector_in_face_reference_frame = self.get_points_in_face_reference_frame(self.barycenter_vector)

        # print(
        #     "face_barycenter_vector_in_face_reference_frame : \n{}\n".format(
        #         face_barycenter_vector_in_face_reference_frame
        #     )
        # )
        # print(
        #     "face_vertices_matrix_in_face_reference_frame : \n{}\n".format(face_vertices_matrix_in_face_reference_frame)
        # )

        # --------------------------------------------------------------------------------------------------------------
        # Getting the face partition into simplices
        # --------------------------------------------------------------------------------------------------------------

        sub_faces_vertices_matrices = self.get_face_partition(
            face_vertices_matrix_in_face_reference_frame, face_barycenter_vector_in_face_reference_frame
        )

        # --------------------------------------------------------------------------------------------------------------
        # Getting face quadrature points and weights in the face hyperplane
        # --------------------------------------------------------------------------------------------------------------

        face_quadrature_points_in_face_reference_frame, face_quadrature_weights = [], []
        face_volume = 0.0
        for sub_face_vertices_matrix in sub_faces_vertices_matrices:
            # print("sub_face_vertices_matrix : \n{}\n".format(sub_face_vertices_matrix))
            sub_face_volume = self.get_simplex_volume(sub_face_vertices_matrix)
            # print("sub_face_volume : \n{}\n".format(sub_face_volume))
            face_volume += sub_face_volume
            (
                sub_face_quadrature_points_in_face_reference_frame,
                sub_face_quadrature_weights,
            ) = self.get_simplex_quadrature(sub_face_vertices_matrix, sub_face_volume, k)

            face_quadrature_points_in_face_reference_frame.append(sub_face_quadrature_points_in_face_reference_frame)
            face_quadrature_weights.append(sub_face_quadrature_weights)
        face_quadrature_points_in_face_reference_frame = np.concatenate(
            face_quadrature_points_in_face_reference_frame, axis=0
        )
        face_quadrature_weights = np.concatenate(face_quadrature_weights, axis=0)
        self.volume = face_volume

        # --------------------------------------------------------------------------------------------------------------
        # Retrieving the face quadrature points in the cell dimension
        # --------------------------------------------------------------------------------------------------------------

        number_of_quadrature_points = face_quadrature_points_in_face_reference_frame.shape[0]
        distance_to_origin = self.get_face_distance_to_origin()
        face_distance_to_origin_vector = np.full((number_of_quadrature_points, 1), distance_to_origin)
        face_quadrature_points_in_face_reference_frame = np.concatenate(
            (face_quadrature_points_in_face_reference_frame, face_distance_to_origin_vector), axis=1
        )

        # --------------------------------------------------------------------------------------------------------------
        # Inverting the mapping matrix from the cell reference frame to the face reference frame
        # --------------------------------------------------------------------------------------------------------------

        p_inv = np.linalg.inv(self.reference_frame_transformation_matrix)

        # --------------------------------------------------------------------------------------------------------------
        # Retrieving the face quadrature points in the cell reference frame
        # --------------------------------------------------------------------------------------------------------------
        # print(face_quadrature_points_in_face_reference_frame)
        # print(p_inv)
        self.quadrature_points = (p_inv @ face_quadrature_points_in_face_reference_frame.T).T
        self.quadrature_weights = face_quadrature_weights

    def get_face_reference_frame_transformation_matrix(self, face_vertices_matrix: Mat) -> Mat:
        """
        """
        d = face_vertices_matrix.shape[1]
        # --------------------------------------------------------------------------------------------------------------
        # 2d faces in 3d cells
        # --------------------------------------------------------------------------------------------------------------
        if d == 3:
            # self.barycenter_vector[0] --> [0] parce que c'est une matrice 3-1
            e_0 = face_vertices_matrix[0] - self.barycenter_vector[0]
            e_0 = e_0 / np.linalg.norm(e_0)
            e_test = face_vertices_matrix[1] - self.barycenter_vector[0]
            e_2 = np.cross(e_0, e_test)
            e_2 = e_2 / np.linalg.norm(e_2)
            e_1 = np.cross(e_2, e_0)
            face_reference_frame_transformation_matrix = np.array([e_0, e_1, e_2])
        # --------------------------------------------------------------------------------------------------------------
        # 1d faces in 2d cells
        # --------------------------------------------------------------------------------------------------------------
        elif d == 2:
            e_0 = face_vertices_matrix[1, :] - face_vertices_matrix[0, :]
            e_0 = e_0 / np.linalg.norm(e_0)
            e_1 = np.array([e_0[1], -e_0[0]])
            face_reference_frame_transformation_matrix = np.array([e_0, e_1])
        # --------------------------------------------------------------------------------------------------------------
        # 0d faces in 1d cells
        # --------------------------------------------------------------------------------------------------------------
        elif d == 1:
            face_reference_frame_transformation_matrix = np.array([[1.0]])
        return face_reference_frame_transformation_matrix

    def get_face_normal_vector(self) -> Mat:
        """
        """
        p = self.reference_frame_transformation_matrix
        return p[-1]

    def get_face_distance_to_origin(self) -> float:
        """
        """
        p = self.reference_frame_transformation_matrix
        # [:, -1][0] --> un  point est comme ca [[1,2,3]] donc on prend le dernier element
        return ((p @ self.barycenter_vector.T).T)[:, -1][0]

    def get_points_in_face_reference_frame(self, points_matrix):
        """
        """
        d = self.reference_frame_transformation_matrix.shape[1]
        if d == 1 or d == 2:
            cols = [0]
        if d == 3:
            cols = [0, 1]
        p = self.reference_frame_transformation_matrix
        return ((p @ points_matrix.T).T)[:, cols]

    def get_face_partition(
        self, face_vertices_matrix_in_face_reference_frame: Mat, face_barycenter_vector_in_face_reference_frame: Mat
    ) -> Mat:
        """
        """
        d = face_vertices_matrix_in_face_reference_frame.shape[1]
        # --------------------------------------------------------------------------------------------------------------
        # Reading the problem dimension
        # --------------------------------------------------------------------------------------------------------------
        if d == 1:
            simplicial_sub_faces = [face_vertices_matrix_in_face_reference_frame]
        if d == 2:
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
