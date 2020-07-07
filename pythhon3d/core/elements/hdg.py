from core.face import Face
from core.cell import Cell
from core.operators import Operators

# from core.operators.operator import Operator
# from core.operators.gradient import Gradient
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class HdgElement:
    """
    ====================================================================================================================
    Class :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Attributes :
    ====================================================================================================================
    
    """

    def __init__(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        # direction: int,
        derivative_direction: int,
    ):
        b_faces = []
        z_faces = []
        # --------------------------------------------------------------------------------------------------------------
        m_cell_mass_left = Operators.get_cell_mass_matrix_in_cell(cell, cell_basis)
        m_cell_advc_right = Operators.get_cell_advection_matrix_in_cell(cell, cell_basis, derivative_direction)
        b_cell = np.copy(m_cell_advc_right)
        z_cell = np.zeros((face_basis.basis_dimension, cell_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for face in faces:
            # ----------------------------------------------------------------------------------------------------------
            # Getting the face orientation with respect to the cell
            # ----------------------------------------------------------------------------------------------------------
            vector_to_face = self.get_vector_to_face(cell, face)
            if vector_to_face[-1] > 0:
                n = -1
                passmat = self.get_swaped_face_reference_frame_transformation_matrix(cell, face)
            else:
                n = 1
                passmat = face.reference_frame_transformation_matrix
            # ----------------------------------------------------------------------------------------------------------
            # Getting the face orientation with respect to the cell
            # ----------------------------------------------------------------------------------------------------------
            m_cell_mass_right = Operators.get_cell_mass_matrix_in_face(cell, face, cell_basis)
            m_hybr_mass_right = Operators.get_hybrid_mass_matrix_in_face(cell, face, cell_basis, face_basis, passmat)
            # ------------------------------------------------------------------------------------------------------
            # b_cell = m_cell_advc_right - m_cell_mass_right
            b_cell -= m_cell_mass_right
            # ------------------------------------------------------------------------------------------------------
            face_normal_vector = passmat[-1]
            b_face = m_hybr_mass_right * face_normal_vector[derivative_direction]
            # ------------------------------------------------------------------------------------------------------
            b_faces.append(b_face)
            # ------------------------------------------------------------------------------------------------------
            # passmats.append(passmat)
            face_mass_matrix_in_face = Operators.get_face_mass_matrix_in_face(face, face_basis, passmat)
            z_cell -= (np.linalg.inv(face_mass_matrix_in_face)) @ (m_hybr_mass_right.T)
            z_face = np.ones((face_basis.basis_dimension, face_basis.basis_dimension))
            # ------------------------------------------------------------------------------------------------------
            z_faces.append(z_face)
        # ----------------------------------------------------------------------------------------------------------
        b_rights = [b_cell] + b_faces
        b_right = np.concatenate(b_rights, axis=1)
        self.reconstructed_gradient_operator = np.linalg.inv(m_cell_mass_left) @ b_right
        print(self.reconstructed_gradient_operator)
        # --------------------------------------------------------------------------------------------------------------
        z = [z_cell] + z_faces
        self.stabilization = np.concatenate(z, axis=1)
        print(self.stabilization)

    def get_vector_to_face(self, cell: Cell, face: Face) -> Mat:
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
        p = face.reference_frame_transformation_matrix
        vector_to_face = (p @ (cell.centroid - face.centroid).T).T
        return vector_to_face

    def get_swaped_face_reference_frame_transformation_matrix(self, cell: Cell, face: Face) -> Mat:
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
