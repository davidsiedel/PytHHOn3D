from core.face import Face
from core.cell import Cell

# from bases.basis import Basis
# from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Operator:
    def __init__(
        self,
        local_cell_mass_matrix: Mat,
        # local_face_mass_matrix: Mat,
        local_identity_operator: Mat,
        local_reconstructed_gradient_operators: List[Mat],
        local_stabilization_matrix: Mat,
        local_load_vectors: Mat,
        local_pressure_vectors: Mat,
    ):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Operator class provides general a framework to build an non-confromal element. It is built using both the
        local gradient operator and the stabilization operator that are used in non-conformal methods to define the local equilibrium of a cell.
        The local gradient is used in place of the regular gradient operator (see e.g. the Basis class) to write a
        behavior law.
        The stabilization operator is used to define a stabilization force that relates the cell to faces, in order to
        recover a weak form of regularity.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - local_gradient_operators : the local gradient operator for the given element
        - local_stabilization_matrix : the local stabilization operator for the given element
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - local_gradient_operators : the local gradient operator for the given element
        - local_stabilization_matrix : the local stabilization operator for the given element
        """
        self.local_cell_mass_matrix = local_cell_mass_matrix
        # self.local_face_mass_matrix = local_face_mass_matrix
        self.local_identity_operator = local_identity_operator
        self.local_reconstructed_gradient_operators = local_reconstructed_gradient_operators
        self.local_stabilization_matrix = local_stabilization_matrix
        self.local_load_vectors = local_load_vectors
        self.local_pressure_vectors = local_pressure_vectors

    def get_vector_to_face(self, cell: Cell, face: Face) -> Mat:
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

    def get_swaped_face_reference_frame_transformation_matrix(self, cell: Cell, face: Face) -> Mat:
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
            swaped_reference_frame_transformation_matrix = np.array([-p[0], p[1], -p[2]])
        # --------------------------------------------------------------------------------------------------------------
        # 1d faces in 2d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 2:
            swaped_reference_frame_transformation_matrix = np.array([-p[0], -p[1]])
        # --------------------------------------------------------------------------------------------------------------
        # 0d faces in 1d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 1:
            swaped_reference_frame_transformation_matrix = p
        return swaped_reference_frame_transformation_matrix
