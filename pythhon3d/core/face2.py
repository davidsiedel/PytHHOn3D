import numpy as np
from numpy import ndarray as Mat
from typing import List

from shapes.domain import Domain

from shapes.point import Point
from shapes.segment import Segment
from shapes.triangle import Triangle
from shapes.polygon import Polygon


# from core.domain import Domain
# from core.quadrature import Quadrature
# from core.boundary import Boundary


class Face(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
        """
        en dim 1 : [[p0x]]
        en dim 2 : [[p0x, p0y], [p1x, p1y]]
        en dim 3 : [[p0x, p0y, p0z], [p1x, p1y, p1z], [p2x, p2y, p2z]]
        """
        face_shape = Face.get_face_shape(vertices)
        # --------------------------------------------------------------------------------------------------------------
        # Computing the mapping from the cell reference frame into the face hyperplane
        # --------------------------------------------------------------------------------------------------------------
        self.reference_frame_transformation_matrix = self.get_face_reference_frame_transformation_matrix(vertices)
        # --------------------------------------------------------------------------------------------------------------
        # Computing the mapping from the cell reference frame into the face hyperplane
        # --------------------------------------------------------------------------------------------------------------
        vertices_in_face_reference_frame = self.get_points_in_face_reference_frame(vertices)
        # --------------------------------------------------------------------------------------------------------------
        # Getting integration points and face data
        # --------------------------------------------------------------------------------------------------------------
        if face_shape == "POINT":
            f = Point(vertices_in_face_reference_frame)
            centroid = f.centroid
            volume = f.volume
            quadrature_nodes = f.quadrature_nodes
            quadrature_weights = f.quadrature_weights
            # super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)
            del f
        if face_shape == "SEGMENT":
            f = Segment(vertices_in_face_reference_frame, polynomial_order)
            centroid = f.centroid
            volume = f.volume
            quadrature_nodes = f.quadrature_nodes
            quadrature_weights = f.quadrature_weights
            # super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)
            del f
        if face_shape == "TRIANGLE":
            f = Triangle(vertices_in_face_reference_frame, polynomial_order)
            centroid = f.centroid
            volume = f.volume
            quadrature_nodes = f.quadrature_nodes
            quadrature_weights = f.quadrature_weights
            # super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)
            del f
        if face_shape == "POLYGON":
            f = Polygon(vertices_in_face_reference_frame, polynomial_order)
            centroid = f.centroid
            volume = f.volume
            quadrature_nodes = f.quadrature_nodes
            quadrature_weights = f.quadrature_weights
            # super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)
            del f
        # --------------------------------------------------------------------------------------------------------------
        # Computing the normal component
        # --------------------------------------------------------------------------------------------------------------
        print(self.reference_frame_transformation_matrix)
        distance_to_origin = self.get_face_distance_to_origin(vertices)
        # --------------------------------------------------------------------------------------------------------------
        # Appending the normal component to quadrature points
        # --------------------------------------------------------------------------------------------------------------
        number_of_quadrature_points = quadrature_nodes.shape[0]
        print(number_of_quadrature_points)
        print("distorigin : {}".format(distance_to_origin))
        face_distance_to_origin_vector = np.full((number_of_quadrature_points, 1), distance_to_origin)
        print(face_distance_to_origin_vector)
        quadrature_points_in_face_reference_frame = np.concatenate(
            (quadrature_nodes, face_distance_to_origin_vector), axis=1
        )
        print(quadrature_points_in_face_reference_frame)
        # --------------------------------------------------------------------------------------------------------------
        # Appending the normal component to the centroid
        # --------------------------------------------------------------------------------------------------------------
        face_distance_to_origin_vector = np.full((1,), distance_to_origin)
        centroid_in_face_reference_frame = np.concatenate((centroid, face_distance_to_origin_vector))
        print(centroid_in_face_reference_frame)
        # --------------------------------------------------------------------------------------------------------------
        # Inverting the mapping matrix from the cell reference frame to the face reference frame
        # --------------------------------------------------------------------------------------------------------------
        p_inv = np.linalg.inv(self.reference_frame_transformation_matrix)
        # --------------------------------------------------------------------------------------------------------------
        # Getting the quadrature points and nodes in the cell reference frame
        # --------------------------------------------------------------------------------------------------------------
        quadrature_points = (p_inv @ quadrature_points_in_face_reference_frame.T).T
        quadrature_weights = quadrature_weights
        centroid = (p_inv @ centroid_in_face_reference_frame.T).T
        print("centroid : {}".format(centroid))
        print("centroid : {}".format(centroid.shape))
        super().__init__(centroid, volume, quadrature_points, quadrature_weights)
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"

        # # super().__init__(face_vertices_matrix, polynomial_order)

        # # print("face_vertices_matrix : \n{}\n".format(face_vertices_matrix))

        # # --------------------------------------------------------------------------------------------------------------
        # # Computing the mapping from the cell reference frame into the face hyperplane
        # # --------------------------------------------------------------------------------------------------------------

        # self.reference_frame_transformation_matrix = self.get_face_reference_frame_transformation_matrix(
        #     face_vertices_matrix
        # )
        # # print("reference_frame_transformation_matrix : \n{}\n".format(self.reference_frame_transformation_matrix))

        # # --------------------------------------------------------------------------------------------------------------
        # # Working in face dimension : projecting vertices and barycenter into the face hyperplane
        # # --------------------------------------------------------------------------------------------------------------

        # face_vertices_matrix_in_face_reference_frame = self.get_points_in_face_reference_frame(face_vertices_matrix)
        # face_barycenter_vector_in_face_reference_frame = self.get_points_in_face_reference_frame(self.barycenter_vector)

        # # print(
        # #     "face_barycenter_vector_in_face_reference_frame : \n{}\n".format(
        # #         face_barycenter_vector_in_face_reference_frame
        # #     )
        # # )
        # # print(
        # #     "face_vertices_matrix_in_face_reference_frame : \n{}\n".format(face_vertices_matrix_in_face_reference_frame)
        # # )

        # # --------------------------------------------------------------------------------------------------------------
        # # Getting the face partition into simplices
        # # --------------------------------------------------------------------------------------------------------------

        # sub_faces_vertices_matrices = self.get_face_partition(
        #     face_vertices_matrix_in_face_reference_frame, face_barycenter_vector_in_face_reference_frame
        # )

        # # --------------------------------------------------------------------------------------------------------------
        # # Getting face quadrature points and weights in the face hyperplane
        # # --------------------------------------------------------------------------------------------------------------

        # face_quadrature_points_in_face_reference_frame, face_quadrature_weights = [], []
        # face_volume = 0.0
        # for sub_face_vertices_matrix in sub_faces_vertices_matrices:
        #     # print("sub_face_vertices_matrix : \n{}\n".format(sub_face_vertices_matrix))
        #     sub_face_volume = self.get_simplex_volume(sub_face_vertices_matrix)
        #     # print("sub_face_volume : \n{}\n".format(sub_face_volume))
        #     face_volume += sub_face_volume
        #     (
        #         sub_face_quadrature_points_in_face_reference_frame,
        #         sub_face_quadrature_weights,
        #     ) = self.get_simplex_quadrature(sub_face_vertices_matrix, sub_face_volume, k)

        #     face_quadrature_points_in_face_reference_frame.append(sub_face_quadrature_points_in_face_reference_frame)
        #     face_quadrature_weights.append(sub_face_quadrature_weights)
        # face_quadrature_points_in_face_reference_frame = np.concatenate(
        #     face_quadrature_points_in_face_reference_frame, axis=0
        # )
        # face_quadrature_weights = np.concatenate(face_quadrature_weights, axis=0)
        # self.volume = face_volume

        # # --------------------------------------------------------------------------------------------------------------
        # # Retrieving the face quadrature points in the cell dimension
        # # --------------------------------------------------------------------------------------------------------------

        # number_of_quadrature_points = face_quadrature_points_in_face_reference_frame.shape[0]
        # distance_to_origin = self.get_face_distance_to_origin()
        # face_distance_to_origin_vector = np.full((number_of_quadrature_points, 1), distance_to_origin)
        # face_quadrature_points_in_face_reference_frame = np.concatenate(
        #     (face_quadrature_points_in_face_reference_frame, face_distance_to_origin_vector), axis=1
        # )

        # # --------------------------------------------------------------------------------------------------------------
        # # Inverting the mapping matrix from the cell reference frame to the face reference frame
        # # --------------------------------------------------------------------------------------------------------------

        # p_inv = np.linalg.inv(self.reference_frame_transformation_matrix)

        # # --------------------------------------------------------------------------------------------------------------
        # # Retrieving the face quadrature points in the cell reference frame
        # # --------------------------------------------------------------------------------------------------------------
        # # print(face_quadrature_points_in_face_reference_frame)
        # # print(p_inv)
        # self.quadrature_points = (p_inv @ face_quadrature_points_in_face_reference_frame.T).T
        # self.quadrature_weights = face_quadrature_weights
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"

    @staticmethod
    def get_face_shape(vertices: Mat) -> str:
        number_of_vertices = vertices.shape[0]
        problem_dimension = vertices.shape[1]
        if number_of_vertices == 1 and problem_dimension == 1:
            face_shape = "POINT"
        if number_of_vertices == 2 and problem_dimension == 2:
            face_shape = "SEGMENT"
        if number_of_vertices == 3 and problem_dimension == 3:
            face_shape = "TRIANGLE"
        if number_of_vertices == 4 and problem_dimension == 3:
            face_shape = "QUADRANGLE"
        if number_of_vertices > 4 and problem_dimension == 3:
            face_shape = "POLYGON"
        return face_shape

    def get_face_reference_frame_transformation_matrix(self, face_vertices_matrix: Mat) -> Mat:
        """
        """
        d = face_vertices_matrix.shape[1]
        # --------------------------------------------------------------------------------------------------------------
        # 2d faces in 3d cells
        # --------------------------------------------------------------------------------------------------------------
        if d == 3:
            # self.barycenter_vector[0] --> [0] parce que c'est une matrice 3-1
            # e_0 = face_vertices_matrix[0] - self.barycenter_vector[0]
            e_0 = face_vertices_matrix[0] - face_vertices_matrix[-1]
            e_0 = e_0 / np.linalg.norm(e_0)
            # e_test = face_vertices_matrix[1] - self.barycenter_vector[0]
            e_test = face_vertices_matrix[1] - face_vertices_matrix[-1]
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

    def get_face_distance_to_origin(self, vertices) -> float:
        """
        """
        p = self.reference_frame_transformation_matrix
        # [:, -1][0] --> un  point est comme ca [[1,2,3]] donc on prend le dernier element
        return ((p @ vertices.T).T)[0, -1]

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

    # @staticmethod
    # def get_face_partition(
    #     face_vertices_matrix_in_face_reference_frame: Mat, face_barycenter_vector_in_face_reference_frame: Mat
    # ) -> Mat:
    #     """
    #     """
    #     d = face_vertices_matrix_in_face_reference_frame.shape[1]
    #     # --------------------------------------------------------------------------------------------------------------
    #     # Reading the problem dimension
    #     # --------------------------------------------------------------------------------------------------------------
    #     if d == 1:
    #         simplicial_sub_faces = [face_vertices_matrix_in_face_reference_frame]
    #     if d == 2:
    #         # ----------------------------------------------------------------------------------------------------------
    #         # getting the number of vertices.
    #         # ----------------------------------------------------------------------------
    #         number_of_vertices = face_vertices_matrix_in_face_reference_frame.shape[0]
    #         # ----------------------------------------------------------------------------
    #         # If the face is no simplex (i.e. of the number of vertices is greater than
    #         # (d-1)+1 = d), proceed to its segmentation in simplices.
    #         # ----------------------------------------------------------------------------
    #         if number_of_vertices > d + 1:
    #             simplicial_sub_faces = []
    #             for i in range(number_of_vertices):
    #                 sub_face_vertices_matrix = [
    #                     face_vertices_matrix_in_face_reference_frame[i - 1, :],
    #                     face_vertices_matrix_in_face_reference_frame[i, :],
    #                     face_barycenter_vector_in_face_reference_frame[0],
    #                 ]
    #                 simplicial_sub_faces.append(np.array(sub_face_vertices_matrix))
    #         else:
    #             simplicial_sub_faces = [face_vertices_matrix_in_face_reference_frame]
    #     return simplicial_sub_faces
