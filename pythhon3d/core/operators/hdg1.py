from core.face import Face
from core.cell import Cell
from core.integration import Integration
from core.operators.operator import Operator
from bases.basis import Basis
from bases.monomial import ScaledMonomial
from behaviors.behavior import Behavior
from behaviors.laplacian import Laplacian

import numpy as np
from typing import List
from numpy import ndarray as Mat


class HDG(Operator):
    def __init__(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        problem_dimension: int,
        behavior: Behavior,
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
        for i, face in enumerate(faces):
            vector_to_face = self.get_vector_to_face(cell, face)
            if vector_to_face[-1] > 0:
                n = -1
                passmat = self.get_swaped_face_reference_frame_transformation_matrix(cell, face)
            else:
                n = 1
                passmat = face.reference_frame_transformation_matrix

        super().__init__(
            m_cell_mass_left,
            # face_mass_matrix_in_face,
            local_identity_operator,
            local_reconstructed_gradient_operators,
            local_stabilization_matrix,
            local_load_vectors,
            local_pressure_vectors,
        )

    def get_local_descrete_gradient_right_hand_side_components(
        self,
        face: Face,
        cell: Cell,
        face_basis: Basis,
        cell_basis: Basis,
        problem_dimension: int,
        derivative_direction: int,
    ):

        return

