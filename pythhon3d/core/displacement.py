import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from core.cell import Cell
from core.face import Face
from bases.basis import Basis
from core.integration import Integration
from core.unknown import Unknown


class Displacement:
    def __init__(
        self,
        face: Face,
        face_basis: Basis,
        face_reference_frame_transformation_matrix: Mat,
        # unknown: Unknown,
        displacement: List[Callable],
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
        displacement_vector = Integration.get_face_pressure_vector_in_face(
            face, face_basis, face_reference_frame_transformation_matrix, displacement
        )
        self.displacement_vector = displacement_vector

    #     displacement_vector = np.zeros((face_basis.basis_dimension,))
    #     for i, displacement_component in enumerate(displacement):
    #         if displacement_component is None:
    #             displacement_vector_component = np.zeros((face_basis.basis_dimension,))
    #         else:
    #             # face: Face, face_basis: Basis, face_reference_frame_transformation_matrix: Mat, displacement: Callable,
    #             displacement_vector_component = Integration.get_face_pressure_vector_in_face(
    #                 face, face_basis, face_reference_frame_transformation_matrix, displacement_component
    #             )
    #         displacement_vector[
    #             i * face_basis.basis_dimension : (i + 1) * face_basis.basis_dimension
    #         ] += displacement_vector_component

    # self.displacement_vector = displacement_vector

    def check_displacement_consistency(self, displacement: List[Callable], unknown: Unknown):
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
        if not len(displacement) == unknown.field_dimension:
            raise ValueError("Attention")
