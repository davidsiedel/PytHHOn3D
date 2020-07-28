from core.face import Face
from core.cell import Cell
from core.integration import Integration
from core.operators.operator import Operator
from core.unknown import Unknown
from core.element import Element
from core.load import Load
from core.pressure import Pressure
from bases.basis import Basis
from bases.monomial import ScaledMonomial
from behaviors.behavior import Behavior
from behaviors.laplacian import Laplacian

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Problem:
    def __init__(
        self, element: Element, external_forces: Mat, tangent_matrix: Mat, cell_basis: Basis, face_basis: Basis,
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
        local_internal_forces = self.get_local_internal_forces(element, tangent_matrix, cell_basis)

    def get_local_internal_forces(self, element: Element, tangent_matrix: Mat, cell_basis: Basis) -> Mat:
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
        # --------------------------------------------------------------------------------------------------------------
        # Stiffness form
        # --------------------------------------------------------------------------------------------------------------
        dim_c = cell_basis.basis_dimension
        tangent_matrix_size = tangent_matrix.shape[0]
        total_size = tangent_matrix_size * dim_c
        local_mass_matrix = np.zeros((total_size, total_size))
        for i in range(tangent_matrix.shape[0]):
            for j in range(tangent_matrix.shape[1]):
                m_phi_phi_cell = Integration.get_cell_tangent_matrix_in_cell(cell, cell_basis, behavior)
                m = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)
                # ------------------------------------------------------------------------------------------------------
                l0 = i * dim_c
                l1 = (i + 1) * dim_c
                c0 = j * dim_c
                c1 = (j + 1) * dim_c
                # ------------------------------------------------------------------------------------------------------
                local_mass_matrix[l0:l1, c0:c1] += tangent_matrix[i, j] * m_phi_phi_cell
        local_gradient_operator = element.local_gradient_operator
        local_stiffness_form = local_gradient_operator.T @ local_mass_matrix @ local_gradient_operator
        # --------------------------------------------------------------------------------------------------------------
        # Stabilization form
        # --------------------------------------------------------------------------------------------------------------
        local_stabilization_form = element.local_stabilization_form
        # --------------------------------------------------------------------------------------------------------------
        # Bilinear form
        # --------------------------------------------------------------------------------------------------------------
        local_internal_forces = local_stiffness_form + local_stabilization_form
        return local_internal_forces
