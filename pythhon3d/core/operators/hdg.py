from core.face import Face
from core.cell import Cell
from core.integration import Integration
from core.operators.operator import Operator
from core.unknown import Unknown
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class HDG(Operator):
    def __init__(self, cell: Cell, faces: List[Face], cell_basis: Basis, face_basis: Basis, unknown: Unknown):
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
        # --------------------------------------------------------------------------------------------------------------
        # Getting the local problme size depending on the problem dimension and the field dimension
        # --------------------------------------------------------------------------------------------------------------
        local_problem_size = self.get_local_problem_size(faces, cell_basis, face_basis, unknown)
        # --------------------------------------------------------------------------------------------------------------
        # Initializing the local gradient operator matrix
        # --------------------------------------------------------------------------------------------------------------
        # gradient_output_dimension = len(unknown.indices) * cell_basis.basis_dimension
        # gradient_input_dimension = local_problem_size
        # local_gradient_operator = np.zeros((gradient_input_dimension, gradient_output_dimension))
        local_gradient_operator = np.zeros((len(unknown.indices) * cell_basis.basis_dimension, local_problem_size))
        # --------------------------------------------------------------------------------------------------------------
        # Initializing the local stabilization form matrix
        # --------------------------------------------------------------------------------------------------------------
        local_stabilization_form = np.zeros((local_problem_size, local_problem_size))
        local_mass_operator = np.zeros((local_problem_size, local_problem_size))
        # --------------------------------------------------------------------------------------------------------------
        # Computing cell mass matrices in cell
        # --------------------------------------------------------------------------------------------------------------
        m_phi_phi_cell = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)
        m_phi_phi_cell_inv = np.linalg.inv(m_phi_phi_cell)
        # --------------------------------------------------------------------------------------------------------------
        # Listing the field directions and derivative directions
        # --------------------------------------------------------------------------------------------------------------
        derivative_directions = range(unknown.problem_dimension)
        directions = range(unknown.field_dimension)
        for j in derivative_directions:
            m_phi_grad_phi_cell = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis, j)
            for i in directions:
                # ------------------------------------------------------------------------------------------------------
                # Getting the line in the gradient operator for the derivative of the field on the ith axis along the
                # jth direction
                # ------------------------------------------------------------------------------------------------------
                line = self.get_line_from_indices(i, j, unknown)
                l0 = line * cell_basis.basis_dimension
                l1 = (line + 1) * cell_basis.basis_dimension
                # ------------------------------------------------------------------------------------------------------
                # Getting the indices corresponding to the cell for the field on the ith axis
                # ------------------------------------------------------------------------------------------------------
                c0 = i * cell_basis.basis_dimension
                c1 = (i + 1) * cell_basis.basis_dimension
                # ------------------------------------------------------------------------------------------------------
                # Writing the advection contribution in the gradient operator the field on the ith axis along the jth
                # direction
                # ------------------------------------------------------------------------------------------------------
                local_gradient_operator[l0:l1, c0:c1] += m_phi_grad_phi_cell
                # ------------------------------------------------------------------------------------------------------
                # Writing the mass contribution in the mass operator for the field on the ith axis
                # ------------------------------------------------------------------------------------------------------
                local_mass_operator[c0:c1, c0:c1] += m_phi_phi_cell
                for face_index, face in enumerate(faces):
                    passmat = Operator.get_face_passmat(cell, face)
                    m_phi_phi_face = Integration.get_cell_mass_matrix_in_face(cell, face, cell_basis)
                    m_psi_psi_face = Integration.get_face_mass_matrix_in_face(face, face_basis, passmat)
                    m_psi_psi_face_inv = np.linalg.inv(m_psi_psi_face)
                    m_phi_psi_face = Integration.get_hybrid_mass_matrix_in_face(
                        cell, face, cell_basis, face_basis, passmat
                    )
                    # --------------------------------------------------------------------------------------------------
                    # Getting the indices corresponding to the face_indexth face for the field on the ith axis
                    # --------------------------------------------------------------------------------------------------
                    f0 = (
                        unknown.field_dimension * cell_basis.basis_dimension
                        + face_index * unknown.field_dimension * face_basis.basis_dimension
                        + i * face_basis.basis_dimension
                    )
                    f1 = (
                        unknown.field_dimension * cell_basis.basis_dimension
                        + face_index * unknown.field_dimension * face_basis.basis_dimension
                        + (i + 1) * face_basis.basis_dimension
                    )
                    # --------------------------------------------------------------------------------------------------
                    # Writing the field jump contribution in the gradient operator
                    # --------------------------------------------------------------------------------------------------
                    local_gradient_operator[l0:l1, c0:c1] -= m_phi_phi_face
                    local_gradient_operator[l0:l1, f0:f1] += m_phi_psi_face
                    # --------------------------------------------------------------------------------------------------
                    # Writing the face contribution in the stabilization form
                    # --------------------------------------------------------------------------------------------------
                    m_cell_face_projection = m_psi_psi_face_inv @ m_phi_psi_face.T
                    m_face_id = np.eye(face_basis.basis_dimension)
                    m_stab_face = np.zeros((face_basis.basis_dimension, local_problem_size))
                    m_stab_face[:, c0:c1] -= m_cell_face_projection
                    m_stab_face[:, f0:f1] += m_face_id
                    local_stabilization_form += m_stab_face.T @ m_psi_psi_face_inv @ m_stab_face
        super().__init__(local_gradient_operator, local_stabilization_form, local_mass_operator)

    def get_line_from_indices(self, i: int, j: int, unknown: Unknown) -> int:
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
        for line, index in enumerate(unknown.indices):
            if index[0] == i and index[1] == j:
                return line
            else:
                pass
        raise ValueError("ATtention")
