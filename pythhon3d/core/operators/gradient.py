from core.operators.operator import Operator
from core.face import Face
from core.cell import Cell
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Gradient(Operator):
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
        derivative_direction: int,
        faces_reference_frames_transformation_matrix: List[Mat],
    ):
        super().__init__(
            cell, faces, cell_basis, face_basis, derivative_direction, faces_reference_frames_transformation_matrix
        )

        for face, passmat in (faces, faces_reference_frames_transformation_matrix):
            # ----------------------------------------------------------------------------------------------------------
            # Computing the reconstructed gradient operator matrix
            # ----------------------------------------------------------------------------------------------------------
            m_cell_mass_left = self.get_cell_mass_matrix_in_cell(cell, cell_basis)
            m_cell_advc_right = self.get_cell_advection_matrix_in_cell(cell, cell_basis, derivative_direction)
            m_cell_mass_right = self.get_cell_mass_matrix_in_face(cell, face, cell_basis)
            m_hybr_mass_right = self.get_hybrid_mass_matrix_in_face(cell, face, cell_basis, face_basis)
            # ----------------------------------------------------------------------------------------------------------
            b_cell = m_cell_advc_right - m_cell_mass_right
            # ----------------------------------------------------------------------------------------------------------
            face_normal_vector = passmat[-1]
            b_face = m_hybr_mass_right * face_normal_vector[direction]
            # ----------------------------------------------------------------------------------------------------------
            b_left = np.concatenate((b_cell, b_face), axis=1)
            reconstructed_gradient_operator = np.linalg.inv(m_cell_mass_left) @ b_left
            # ----------------------------------------------------------------------------------------------------------
            # Computing the stabilization operator matrix
            # ----------------------------------------------------------------------------------------------------------
            print("reconstructed_gradient_operator : \n{}".format(reconstructed_gradient_operator))

