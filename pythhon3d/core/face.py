import numpy as np
from numpy import ndarray as Mat
from typing import List

from shapes.domain import Domain

from shapes.point import Point
from shapes.segment import Segment
from shapes.triangle import Triangle
from shapes.polygon import Polygon


class Face(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        
        ================================================================================================================
        Attributes :
        ================================================================================================================
        
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
            del f
        if face_shape == "SEGMENT":
            f = Segment(vertices_in_face_reference_frame, polynomial_order)
            centroid = f.centroid
            volume = f.volume
            quadrature_nodes = f.quadrature_nodes
            quadrature_weights = f.quadrature_weights
            del f
        if face_shape == "TRIANGLE":
            f = Triangle(vertices_in_face_reference_frame, polynomial_order)
            centroid = f.centroid
            volume = f.volume
            quadrature_nodes = f.quadrature_nodes
            quadrature_weights = f.quadrature_weights
            del f
        if face_shape == "POLYGON":
            f = Polygon(vertices_in_face_reference_frame, polynomial_order)
            centroid = f.centroid
            volume = f.volume
            quadrature_nodes = f.quadrature_nodes
            quadrature_weights = f.quadrature_weights
            del f
        # --------------------------------------------------------------------------------------------------------------
        # Computing the normal component
        # --------------------------------------------------------------------------------------------------------------
        distance_to_origin = self.get_face_distance_to_origin(vertices)
        # --------------------------------------------------------------------------------------------------------------
        # Appending the normal component to quadrature points
        # --------------------------------------------------------------------------------------------------------------
        number_of_quadrature_points = quadrature_nodes.shape[0]
        face_distance_to_origin_vector = np.full((number_of_quadrature_points, 1), distance_to_origin)
        quadrature_points_in_face_reference_frame = np.concatenate(
            (quadrature_nodes, face_distance_to_origin_vector), axis=1
        )
        # --------------------------------------------------------------------------------------------------------------
        # Appending the normal component to the centroid
        # --------------------------------------------------------------------------------------------------------------
        face_distance_to_origin_vector = np.full((1,), distance_to_origin)
        centroid_in_face_reference_frame = np.concatenate((centroid, face_distance_to_origin_vector))
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
        # --------------------------------------------------------------------------------------------------------------
        # Building the face domain
        # --------------------------------------------------------------------------------------------------------------
        super().__init__(centroid, volume, quadrature_points, quadrature_weights)

    @staticmethod
    def get_face_shape(vertices: Mat) -> str:
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
        number_of_vertices = vertices.shape[0]
        problem_dimension = vertices.shape[1]
        if number_of_vertices == 1 and problem_dimension == 1:
            face_shape = "POINT"
        elif number_of_vertices == 2 and problem_dimension == 2:
            face_shape = "SEGMENT"
        elif number_of_vertices == 3 and problem_dimension == 3:
            face_shape = "TRIANGLE"
        elif number_of_vertices == 4 and problem_dimension == 3:
            face_shape = "QUADRANGLE"
        elif number_of_vertices > 4 and problem_dimension == 3:
            face_shape = "POLYGON"
        else:
            raise NameError("no match")
        return face_shape

    def get_face_reference_frame_transformation_matrix(self, face_vertices_matrix: Mat) -> Mat:
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
        problem_dimension = face_vertices_matrix.shape[1]
        # --------------------------------------------------------------------------------------------------------------
        # 2d faces in 3d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 3:
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
        elif problem_dimension == 2:
            e_0 = face_vertices_matrix[1, :] - face_vertices_matrix[0, :]
            e_0 = e_0 / np.linalg.norm(e_0)
            e_1 = np.array([e_0[1], -e_0[0]])
            face_reference_frame_transformation_matrix = np.array([e_0, e_1])
        # --------------------------------------------------------------------------------------------------------------
        # 0d faces in 1d cells
        # --------------------------------------------------------------------------------------------------------------
        elif problem_dimension == 1:
            face_reference_frame_transformation_matrix = np.array([[1.0]])
        return face_reference_frame_transformation_matrix

    def get_face_normal_vector(self) -> Mat:
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
        p = self.reference_frame_transformation_matrix
        return p[-1]

    def get_face_distance_to_origin(self, vertices) -> float:
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
        p = self.reference_frame_transformation_matrix
        return ((p @ vertices.T).T)[0, -1]

    def get_points_in_face_reference_frame(self, points_matrix):
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
        problem_dimension = self.reference_frame_transformation_matrix.shape[1]
        if problem_dimension == 1 or problem_dimension == 2:
            cols = [0]
        if problem_dimension == 3:
            cols = [0, 1]
        p = self.reference_frame_transformation_matrix
        return ((p @ points_matrix.T).T)[:, cols]
