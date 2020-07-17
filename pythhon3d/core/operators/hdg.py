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
        The HDG class inherits from the Element class, and builds :
        - a local gradient operator B in P(k,d*d) where k denotes the cell basis dimension and d the spatial dimension
        of the problem. Let u the cell unknown in P(k,d) and v the face unknown in P(k',d-1).
        The local gradient tensor G solves for all T in P(k,d*d):
        G : T = grad(u) : T + (v-u) : T * n
        And the local gradient operator B is then defined such that B * |u,v| = G
        - a local stabilization operator Z that relates u to v at the boundary of the element:
        Z = PI(u) - v
        where PI is the L2-projection operator from P(k,d) onto P(k',d-1)
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - cell : the element cell
        - faces : the element faces
        - cell_basis : the polynomial basis for any cell-like domain
        - face_basis : the polynomial basis for any face-like domain
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - local_gradient_operators : the local gradient operators Bj for the HDG element, where j denotes the derivative
        direction
        - local_stabilization_matrix : the local stabilization operator Z for the HDG element
        """
        # --------------------------------------------------------------------------------------------------------------
        # LOAD
        # --------------------------------------------------------------------------------------------------------------
        if not cell.load is None:
            local_load_vectors = []
            for direction in range(field_dimension):
                if not cell.load[direction] is None:
                    load_vector = Integration.get_cell_load_vector_in_cell(cell, cell_basis, cell.load[direction])
                else:
                    load_vector = np.zeros((cell_basis.basis_dimension,))
                local_load_vectors.append(load_vector)
        else:
            local_load_vectors = [np.zeros((cell_basis.basis_dimension,)) for i in range(field_dimension)]
        # --------------------------------------------------------------------------------------------------------------
        # OPERATORS
        # --------------------------------------------------------------------------------------------------------------
        number_of_faces = len(faces)
        # --------------------------------------------------------------------------------------------------------------
        local_stabilization_matrix = np.zeros(
            (
                cell_basis.basis_dimension + number_of_faces * face_basis.basis_dimension,
                cell_basis.basis_dimension + number_of_faces * face_basis.basis_dimension,
            )
        )
        local_reconstructed_gradient_operators = []
        local_pressure_vectors = []
        # --------------------------------------------------------------------------------------------------------------
        z_cell = np.zeros((face_basis.basis_dimension, cell_basis.basis_dimension))
        z_faces = []
        for derivative_direction in range(problem_dimension):
            # ----------------------------------------------------------------------------------------------------------
            b_faces = []
            # ----------------------------------------------------------------------------------------------------------
            m_cell_mass_left = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)
            m_cell_advc_right = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis, derivative_direction)
            b_cell = np.copy(m_cell_advc_right)
            # ----------------------------------------------------------------------------------------------------------
            for face_index, face in enumerate(faces):
                # ------------------------------------------------------------------------------------------------------
                # Getting the face orientation with respect to the cell
                # ------------------------------------------------------------------------------------------------------
                vector_to_face = self.get_vector_to_face(cell, face)
                if vector_to_face[-1] > 0:
                    n = -1
                    passmat = self.get_swaped_face_reference_frame_transformation_matrix(cell, face)
                else:
                    n = 1
                    passmat = face.reference_frame_transformation_matrix
                # ------------------------------------------------------------------------------------------------------
                m_cell_mass_right = Integration.get_cell_mass_matrix_in_face(cell, face, cell_basis)
                m_hybr_mass_right = Integration.get_hybrid_mass_matrix_in_face(
                    cell, face, cell_basis, face_basis, passmat
                )
                # ------------------------------------------------------------------------------------------------------
                b_cell -= m_cell_mass_right
                # ------------------------------------------------------------------------------------------------------
                face_normal_vector = passmat[-1]
                b_face = m_hybr_mass_right * face_normal_vector[derivative_direction]
                # ------------------------------------------------------------------------------------------------------
                b_faces.append(b_face)
                # ------------------------------------------------------------------------------------------------------
                # Only for the first iteration on directions
                # ------------------------------------------------------------------------------------------------------
                if derivative_direction == 0:
                    face_mass_matrix_in_face = Integration.get_face_mass_matrix_in_face(face, face_basis, passmat)
                    z_cell = -1.0 * (np.linalg.inv(face_mass_matrix_in_face)) @ (m_hybr_mass_right.T)
                    z_faces = []
                    for i in range(number_of_faces):
                        if i == face_index:
                            z_face = np.eye(face_basis.basis_dimension)
                        else:
                            z_face = np.zeros((face_basis.basis_dimension, face_basis.basis_dimension))
                        z_faces.append(z_face)
                    z = [z_cell] + z_faces
                    z = np.concatenate(z, axis=1)
                    local_stabilization_matrix += z.T @ (face_mass_matrix_in_face @ z)
                    # --------------------------------------------------------------------------------------------------
                    # local_stabilization_stiffness_operator += z.T @ face_mass_matrix_in_face @ z
                    # --------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------
                # PRESSURE
                # ------------------------------------------------------------------------------------------------------
                if not face.pressure is None:
                    for direction in range(field_dimension):
                        if not face.pressure[direction] is None:
                            pressure_vector = Integration.get_face_pressure_vector_in_face(
                                face, face_basis, passmat, face.pressure[direction]
                            )
                        else:
                            pressure_vector = np.zeros((face_basis.basis_dimension,))
                        local_pressure_vectors.append(pressure_vector)
                else:
                    local_pressure_vectors += [np.zeros((face_basis.basis_dimension,)) for i in range(field_dimension)]
                # local_pressure_vectors.append(face_pressure_vectors)
                # ------------------------------------------------------------------------------------------------------
                # DISPLACEMENT
                # ------------------------------------------------------------------------------------------------------
                if not face.displacement is None:
                    displacement_vectors = []
                    for direction in range(field_dimension):
                        if not face.displacement[direction] is None:
                            displacement_vector = Integration.get_face_displacement_vector_in_face(
                                face, face_basis, passmat, face.displacement[direction]
                            )
                        else:
                            displacement_vector = np.zeros((face_basis.basis_dimension,))
                        displacement_vectors.append(displacement_vector)
                else:
                    displacement_vectors = [np.zeros((face_basis.basis_dimension,)) for i in range(field_dimension)]
            # ----------------------------------------------------------------------------------------------------------
            b_right = [b_cell] + b_faces
            b_right = np.concatenate(b_right, axis=1)
            local_reconstructed_gradient_operator = np.linalg.inv(m_cell_mass_left) @ b_right
            # local_mechanical_stiffness_operator = b.T @ m_cell_mass_left @ b
            # local_reconstructed_gradient_operators.append(local_mechanical_stiffness_operator)
            local_reconstructed_gradient_operators.append(local_reconstructed_gradient_operator)
        # --------------------------------------------------------------------------------------------------------------
        # local mass opertor
        # --------------------------------------------------------------------------------------------------------------
        i_cell = np.eye(cell_basis.basis_dimension)
        i_face = np.zeros((cell_basis.basis_dimension, number_of_faces * face_basis.basis_dimension))
        local_identity_operator = np.concatenate((i_cell, i_face), axis=1)
        # local_mechanical_mass_operator = a3.T @ m_cell_mass_left @ a3
        # --------------------------------------------------------------------------------------------------------------
        # Building the HDG Operator
        # --------------------------------------------------------------------------------------------------------------
        b_vector = local_load_vectors + local_pressure_vectors
        b_vector = np.concatenate(b_vector)
        super().__init__(
            m_cell_mass_left,
            # face_mass_matrix_in_face,
            local_identity_operator,
            local_reconstructed_gradient_operators,
            local_stabilization_matrix,
            local_load_vectors,
            local_pressure_vectors,
        )

    def get_face_discontinuous_gradient_component(self, derivative_direction: int, face: Face):
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
        return

    def get_global_discontinuous_gradient_operator(self):
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
        return

