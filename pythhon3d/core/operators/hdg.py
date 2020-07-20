from core.face import Face
from core.cell import Cell
from core.integration import Integration
from core.operators.operator import Operator
from core.unknown import Unknown
from bases.basis import Basis
from bases.monomial import ScaledMonomial
from behaviors.behavior import Behavior
from behaviors.laplacian import Laplacian

import numpy as np
from typing import List
from numpy import ndarray as Mat


class HDG(Operator):
    def __init__(
        self, cell: Cell, faces: List[Face], cell_basis: Basis, face_basis: Basis, unknown: Unknown, behavior: Behavior,
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
        problem_dimension = unknown.problem_dimension
        field_dimension = unknown.field_dimension
        # --------------------------------------------------------------------------------------------------------------
        # Load
        # --------------------------------------------------------------------------------------------------------------
        local_load_vector = self.get_load_vector(cell, cell_basis, field_dimension)
        # --------------------------------------------------------------------------------------------------------------
        # Pressure
        # --------------------------------------------------------------------------------------------------------------
        local_pressure_vector = self.get_pressure_vector(cell, faces, face_basis, field_dimension)
        # --------------------------------------------------------------------------------------------------------------
        # B
        # --------------------------------------------------------------------------------------------------------------
        local_b_operator = self.get_global_discontinuous_gradient_operator(
            cell, faces, cell_basis, face_basis, problem_dimension, field_dimension, unknown
        )
        # --------------------------------------------------------------------------------------------------------------
        # Stabilization
        # --------------------------------------------------------------------------------------------------------------
        k_z = self.get_global_stabilization_operator(
            cell, faces, cell_basis, face_basis, problem_dimension, field_dimension
        )
        self.local_gradient_opertor = local_b_operator
        self.local_stabilization_matrix = k_z
        self.local_mass_matrix = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)

    def get_load_vector(self, cell: Cell, cell_basis: Basis, field_dimension: int) -> Mat:
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
        local_load_vectors = np.zeros((field_dimension * cell_basis.basis_dimension,))
        if not cell.load is None:
            # local_load_vectors = []
            for direction in range(field_dimension):
                if not cell.load[direction] is None:
                    load_vector = Integration.get_cell_load_vector_in_cell(cell, cell_basis, cell.load[direction])
                else:
                    load_vector = np.zeros((cell_basis.basis_dimension,))
                # local_load_vectors.append(load_vector)
                local_load_vectors[
                    direction * cell_basis.basis_dimension : (direction + 1) * cell_basis.basis_dimension
                ] += load_vector
        # else:
        #     local_load_vectors = [np.zeros((cell_basis.basis_dimension,)) for i in range(field_dimension)]
        return local_load_vectors

    def get_pressure_vector(self, cell: Cell, faces: List[Face], face_basis: Basis, field_dimension: int) -> Mat:
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
        local_pressure_vectors = []
        number_of_faces = len(faces)
        dim_f = face_basis.basis_dimension
        local_pressure_vectors = np.zeros((field_dimension * number_of_faces * dim_f,))
        for i, face in enumerate(faces):
            if not face.pressure is None:
                passmat = self.get_face_passmat(cell, face)
                for direction in range(field_dimension):
                    if not face.pressure[direction] is None:
                        pressure_vector = Integration.get_face_pressure_vector_in_face(
                            face, face_basis, passmat, face.pressure[direction]
                        )
                    else:
                        pressure_vector = np.zeros((dim_f,))
                    # local_pressure_vectors.append(pressure_vector)
                    local_pressure_vectors[
                        (i * field_dimension + direction) * dim_f : (i * field_dimension + direction + 1) * dim_f
                    ] += pressure_vector
            else:
                local_pressure_vectors[i * field_dimension * dim_f : (i + 1) * field_dimension * dim_f] += np.zeros(
                    (field_dimension * dim_f,)
                )
                # local_pressure_vectors += [np.zeros((dim_f,)) for i in range(field_dimension)]
        return local_pressure_vectors

    def get_global_discontinuous_gradient_operators(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        problem_dimension: int,
        field_dimension: int,
    ):
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
        dim_c = cell_basis.basis_dimension
        dim_f = face_basis.basis_dimension
        # --------------------------------------------------------------------------------------------------------------
        # mk,jlmknk nk kn, gi
        # --------------------------------------------------------------------------------------------------------------
        cell_mass_matrix_in_cell = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)
        b_operators = []
        for derivative_direction in range(field_dimension):
            b = np.zeros((dim_c, dim_c + number_of_faces * dim_f))
            # ----------------------------------------------------------------------------------------------------------
            # ejf
            # ----------------------------------------------------------------------------------------------------------
            cell_advection_matrix_in_cell = Integration.get_cell_advection_matrix_in_cell(
                cell, cell_basis, derivative_direction
            )
            b[:, :dim_c] = np.copy(cell_advection_matrix_in_cell)
            # ----------------------------------------------------------------------------------------------------------
            # ejf
            # ----------------------------------------------------------------------------------------------------------
            for i, face in enumerate(faces):
                passmat = self.get_face_passmat(cell, face)
                # ------------------------------------------------------------------------------------------------------
                # Getting the face orientation with respect to the cell
                # ------------------------------------------------------------------------------------------------------
                m_cell_mass_right = Integration.get_cell_mass_matrix_in_face(cell, face, cell_basis)
                m_hybr_mass_right = Integration.get_hybrid_mass_matrix_in_face(
                    cell, face, cell_basis, face_basis, passmat
                )
                # ------------------------------------------------------------------------------------------------------
                b[:, :dim_c] -= m_cell_mass_right
                # ------------------------------------------------------------------------------------------------------
                face_normal_vector = passmat[-1]
                b_face = m_hybr_mass_right * face_normal_vector[derivative_direction]
                # ------------------------------------------------------------------------------------------------------
                b[:, dim_c + i * dim_f : dim_c + (i + 1) * dim_f,] += b_face
            b = np.linalg.inv(cell_mass_matrix_in_cell) @ b
            b_operators.append(b)
        return b_operators

    def get_global_discontinuous_gradient_operator(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        problem_dimension: int,
        field_dimension: int,
        unknown: Unknown,
    ):
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
        b_operators = self.get_global_discontinuous_gradient_operators(
            cell, faces, cell_basis, face_basis, problem_dimension, field_dimension
        )
        cell_mass_matrix_in_cell = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)
        dim_c = cell_basis.basis_dimension
        dim_f = face_basis.basis_dimension
        number_of_faces = len(faces)
        a = dim_c + number_of_faces * dim_f
        b = len(unknown.indices)
        c = a * b
        # global_mass_matrix = np.zeros((c, c))
        global_b_operator = np.zeros((b * dim_c, c))
        for line, index in enumerate(unknown.indices):
            if not unknown.symmetric_gradient:
                i = index[0]
                j = index[1]
                global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += b_operators[j]
            else:
                if i == j:
                    global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += b_operators[j]
                else:
                    global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += 0.5 * b_operators[j]
                    global_b_operator[line * dim_c : (line + 1) * dim_c, j * a : (j + 1) * a] += 0.5 * b_operators[i]
        return global_b_operator

    def get_global_stabilization_operator(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        problem_dimension: int,
        field_dimension: int,
    ):
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
        dim_c = cell_basis.basis_dimension
        dim_f = face_basis.basis_dimension
        # --------------------------------------------------------------------------------------------------------------
        # Initializing the Z operator
        # --------------------------------------------------------------------------------------------------------------
        k_z = np.zeros((dim_c + number_of_faces * dim_f, dim_c + number_of_faces * dim_f))
        for i, face in enumerate(faces):
            z = np.zeros((dim_f, dim_c + number_of_faces * dim_f))
            passmat = self.get_face_passmat(cell, face)
            hybrid_mass_matrix_in_face = Integration.get_hybrid_mass_matrix_in_face(
                cell, face, cell_basis, face_basis, passmat
            )
            face_mass_matrix_in_face = Integration.get_face_mass_matrix_in_face(face, face_basis, passmat)
            z[:, :dim_c] -= (np.linalg.inv(face_mass_matrix_in_face)) @ (hybrid_mass_matrix_in_face.T)
            z[:, dim_c + i * dim_f : dim_c + (i + 1) * dim_f] += np.eye(dim_f)
            k_z += z.T @ (face_mass_matrix_in_face @ z)
        return k_z

