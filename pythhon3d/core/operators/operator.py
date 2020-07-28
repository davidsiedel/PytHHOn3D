from core.face import Face
from core.cell import Cell
from core.integration import Integration
from core.unknown import Unknown
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Operator:
    def __init__(
        self, local_gradient_operator: Mat, local_stabilization_form: Mat, local_mass_operator: Mat,
    ):
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
        
        """
        self.local_gradient_operator = local_gradient_operator
        self.local_stabilization_form = local_stabilization_form
        self.local_mass_operator = local_mass_operator

    def get_local_problem_size(self, faces: List[Face], cell_basis: Basis, face_basis: Basis, unknown: Unknown):
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
        number_of_faces = len(faces)
        local_problem_size = (
            cell_basis.basis_dimension * unknown.field_dimension
            + number_of_faces * face_basis.basis_dimension * unknown.field_dimension
        )
        return local_problem_size

    @staticmethod
    def get_vector_to_face(cell: Cell, face: Face) -> Mat:
        """
        ================================================================================================================
        Description :
        ================================================================================================================
        Returns the vector in the face reference frame that relates the distance between the face to the cell.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - cell : the cell of the element
        - face : a face belonging to the element
        ================================================================================================================
        Exemple :
        ================================================================================================================
        For a triangular face [[0,0,2], [1,0,2], [0,1,2]] in R^3, returns [0,0,2]
        """
        p = face.reference_frame_transformation_matrix
        vector_to_face = (p @ (cell.centroid - face.centroid).T).T
        return vector_to_face

    @staticmethod
    def get_swaped_face_reference_frame_transformation_matrix(cell: Cell, face: Face) -> Mat:
        """
        ================================================================================================================
        Description :
        ================================================================================================================
        Returns the face transformation matrix swaped in a way that the normal vector to the face is outward-oriented:
        since a face is shared by two cells, the transformation from the cell domain towards the face one changes by a
        sign in accordance to the position of the face with regards to the element cell
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - cell : the element cell
        - face : the considered face
        ================================================================================================================
        Exemple :
        ================================================================================================================
        """
        p = face.reference_frame_transformation_matrix
        problem_dimension = p.shape[1]
        # --------------------------------------------------------------------------------------------------------------
        # 2d faces in 3d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 3:
            # swaped_reference_frame_transformation_matrix = np.array([p[1], p[0], -p[2]])
            swaped_reference_frame_transformation_matrix = np.array([p[0], p[1], -p[2]])
        # --------------------------------------------------------------------------------------------------------------
        # 1d faces in 2d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 2:
            swaped_reference_frame_transformation_matrix = np.array([p[0], -p[1]])
        # --------------------------------------------------------------------------------------------------------------
        # 0d faces in 1d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 1:
            swaped_reference_frame_transformation_matrix = p
        return swaped_reference_frame_transformation_matrix

    @staticmethod
    def get_face_passmat(cell: Cell, face: Face):
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
        vector_to_face = Operator.get_vector_to_face(cell, face)
        if vector_to_face[-1] > 0:
            passmat = Operator.get_swaped_face_reference_frame_transformation_matrix(cell, face)
        else:
            passmat = face.reference_frame_transformation_matrix
        return passmat
