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
        self.local_b_vector = np.concatenate((local_load_vector, local_pressure_vector))
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
        self.cell_mass_matrix = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)

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
                print("load_vector : {}".format(load_vector))
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
        # field_direction: int,
        # derivative_direction: int,
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
        b_cells, b_faces = [], []
        for derivative_direction in range(field_dimension):
            b = np.zeros((dim_c, dim_c + number_of_faces * dim_f))
            # ----------------------------------------------------------------------------------------------------------
            # ejf
            # ----------------------------------------------------------------------------------------------------------
            cell_advection_matrix_in_cell = Integration.get_cell_advection_matrix_in_cell(
                cell, cell_basis, derivative_direction
            )
            b[:, :dim_c] = np.copy(cell_advection_matrix_in_cell)
            b_cell = np.copy(cell_advection_matrix_in_cell)
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

    def get_discontinuous_gradient_operator_components(
        self, cell: Cell, faces: List[Face], cell_basis: Basis, face_basis: Basis, derivative_direction: int,
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
        # cell_mass_matrix_in_cell = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)
        # --------------------------------------------------------------------------------------------------------------
        # ejf
        # --------------------------------------------------------------------------------------------------------------
        cell_advection_matrix_in_cell = Integration.get_cell_advection_matrix_in_cell(
            cell, cell_basis, derivative_direction
        )
        b_faces = []
        b_cell = np.copy(cell_advection_matrix_in_cell)
        print("b_cell : \n{}".format(b_cell))
        # --------------------------------------------------------------------------------------------------------------
        # ejf
        # --------------------------------------------------------------------------------------------------------------
        for i, face in enumerate(faces):
            passmat = self.get_face_passmat(cell, face)
            # ----------------------------------------------------------------------------------------------------------
            # Getting the face orientation with respect to the cell
            # ----------------------------------------------------------------------------------------------------------
            m_cell_mass_right = Integration.get_cell_mass_matrix_in_face(cell, face, cell_basis)
            print("m_cell_mass_right : \n{}".format(m_cell_mass_right))
            m_hybr_mass_right = Integration.get_hybrid_mass_matrix_in_face(cell, face, cell_basis, face_basis, passmat)
            print("m_hybr_mass_right : \n{}".format(m_hybr_mass_right))
            # ----------------------------------------------------------------------------------------------------------
            face_normal_vector = passmat[-1]
            # print("face_normal_vector : {}".format(face_normal_vector))
            b_face = m_hybr_mass_right * face_normal_vector[derivative_direction]
            # ----------------------------------------------------------------------------------------------------------
            b_faces.append(b_face)
            b_cell -= m_cell_mass_right
        return b_cell, b_faces

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
        # b_operators = self.get_global_discontinuous_gradient_operators(
        #     cell, faces, cell_basis, face_basis, problem_dimension, field_dimension
        # )
        cell_mass_matrix_in_cell = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis)
        m_inv = np.linalg.inv(cell_mass_matrix_in_cell)
        print("cell_mass_matrix_in_cell : \n{}".format(cell_mass_matrix_in_cell))
        print("m_inv : \n{}".format(m_inv))
        dim_c = cell_basis.basis_dimension
        dim_f = face_basis.basis_dimension
        number_of_faces = len(faces)
        number_of_indices = len(unknown.indices)
        # a = field_dimension * (dim_c + number_of_faces * dim_f)
        cols = field_dimension * (dim_c + number_of_faces * dim_f)
        lines = number_of_indices * dim_c
        global_b_operator = np.zeros((lines, cols))
        b_cell_list, b_faces_list = [], []
        for derivative_direction in range(problem_dimension):
            b_cell, b_faces = self.get_discontinuous_gradient_operator_components(
                cell, faces, cell_basis, face_basis, derivative_direction
            )
            b_cell_list.append(b_cell)
            b_faces_list.append(b_faces)
        for line, index in enumerate(unknown.indices):
            if not unknown.symmetric_gradient:
                i = index[0]
                j = index[1]
                l0 = line * dim_c
                l1 = (line + 1) * dim_c
                c0 = i * dim_c
                c1 = (i + 1) * dim_c
                global_b_operator[l0:l1, c0:c1] += m_inv @ b_cell_list[j]
                for face_index in range(number_of_faces):
                    l0 = line * dim_c
                    l1 = (line + 1) * dim_c
                    c0 = field_dimension * (dim_c + face_index * dim_f) + i * dim_f
                    c1 = field_dimension * (dim_c + face_index * dim_f) + (i + 1) * dim_f
                    global_b_operator[l0:l1, c0:c1] += m_inv @ b_faces_list[j][face_index]
            else:
                i = index[0]
                j = index[1]
                if not i == j:
                    l0 = line * dim_c
                    l1 = (line + 1) * dim_c
                    c0 = i * dim_c
                    c1 = (i + 1) * dim_c
                    global_b_operator[l0:l1, c0:c1] += 0.5 * m_inv @ b_cell_list[j]
                    # --------------------------------------------------------------------------------------------------
                    l0 = line * dim_c
                    l1 = (line + 1) * dim_c
                    c0 = j * dim_c
                    c1 = (j + 1) * dim_c
                    global_b_operator[l0:l1, c0:c1] += 0.5 * m_inv @ b_cell_list[i]
                    # --------------------------------------------------------------------------------------------------
                    for face_index in range(number_of_faces):
                        l0 = line * dim_c
                        l1 = (line + 1) * dim_c
                        c0 = field_dimension * (dim_c + face_index * dim_f) + i * dim_f
                        c1 = field_dimension * (dim_c + face_index * dim_f) + (i + 1) * dim_f
                        global_b_operator[l0:l1, c0:c1] += 0.5 * m_inv @ b_faces_list[j][face_index]
                        # ----------------------------------------------------------------------------------------------
                        l0 = line * dim_c
                        l1 = (line + 1) * dim_c
                        c0 = field_dimension * (dim_c + face_index * dim_f) + j * dim_f
                        c1 = field_dimension * (dim_c + face_index * dim_f) + (j + 1) * dim_f
                        global_b_operator[l0:l1, c0:c1] += 0.5 * m_inv @ b_faces_list[i][face_index]
                        # ----------------------------------------------------------------------------------------------
                else:
                    l0 = line * dim_c
                    l1 = (line + 1) * dim_c
                    c0 = i * dim_c
                    c1 = (i + 1) * dim_c
                    global_b_operator[l0:l1, c0:c1] += m_inv @ b_cell_list[j]
                    for face_index in range(number_of_faces):
                        l0 = line * dim_c
                        l1 = (line + 1) * dim_c
                        c0 = field_dimension * (dim_c + face_index * dim_f) + i * dim_f
                        c1 = field_dimension * (dim_c + face_index * dim_f) + (i + 1) * dim_f
                        global_b_operator[l0:l1, c0:c1] += m_inv @ b_faces_list[j][face_index]
            print("global_b_operator : \n{}".format(global_b_operator))
            print("global_b_operator shape : \n{}".format(global_b_operator.shape))
            return global_b_operator
            #     global_b_operator[
            #         line * dim_c : (line + 1) * dim_c,
            #         field_dimension * dim_c
            #         + (i * face_index) * dim_f : field_dimension * dim_c
            #         + (i * face_index + 1) * dim_f,
            #     ] += b_faces_list[face_index][j]
            # global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += b_cell_list[j]
            # global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += b_operators[j]
            # else:
            #     if i == j:
            #         global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += b_operators[j]
            #     else:
            #         global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += 0.5 * b_operators[j]
            #         global_b_operator[line * dim_c : (line + 1) * dim_c, j * a : (j + 1) * a] += 0.5 * b_operators[i]

        # b = len(unknown.indices)
        # c = a * b
        # global_b_operator = np.zeros((b * dim_c, c))
        # b_cell_list, b_faces_list = [], []
        # for derivative_direction in range(problem_dimension):
        #     b_cell, b_faces = self.get_discontinuous_gradient_operator_components(
        #         cell, faces, cell_basis, face_basis, derivative_direction
        #     )
        #     b_cell_list.append(b_cell)
        #     b_faces_list.append(b_faces)
        # for line, index in enumerate(unknown.indices):
        #     if not unknown.symmetric_gradient:
        #         i = index[0]
        #         j = index[1]
        #         global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += b_operators[j]
        #         global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += b_operators[j]
        #     else:
        #         if i == j:
        #             global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += b_operators[j]
        #         else:
        #             global_b_operator[line * dim_c : (line + 1) * dim_c, i * a : (i + 1) * a] += 0.5 * b_operators[j]
        #             global_b_operator[line * dim_c : (line + 1) * dim_c, j * a : (j + 1) * a] += 0.5 * b_operators[i]
        # return global_b_operator

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
        size_tot = field_dimension * (dim_c + number_of_faces * dim_f)
        # lines = field_dimension * dim_f
        # k_z = np.zeros((dim_c + number_of_faces * dim_f, dim_c + number_of_faces * dim_f))
        k_z = np.zeros((size_tot, size_tot))
        for i, face in enumerate(faces):
            cols = field_dimension * (dim_c + number_of_faces * dim_f)
            lines = dim_f
            # z = np.zeros((dim_f, dim_c + number_of_faces * dim_f))
            z = np.zeros((lines, cols))
            passmat = self.get_face_passmat(cell, face)
            hybrid_mass_matrix_in_face = Integration.get_hybrid_mass_matrix_in_face(
                cell, face, cell_basis, face_basis, passmat
            )
            face_mass_matrix_in_face = Integration.get_face_mass_matrix_in_face(face, face_basis, passmat)
            for direction in range(field_dimension):
                # z[:, :dim_c] -= (np.linalg.inv(face_mass_matrix_in_face)) @ (hybrid_mass_matrix_in_face.T)
                z[:, direction * dim_c : (direction + 1) * dim_c] -= (np.linalg.inv(face_mass_matrix_in_face)) @ (
                    hybrid_mass_matrix_in_face.T
                )
                # z[:, dim_c + i * dim_f : dim_c + (i + 1) * dim_f] += np.eye(dim_f)
                z[
                    :,
                    field_dimension * dim_c
                    + (i + direction) * dim_f : field_dimension * dim_c
                    + (i + direction + 1) * dim_f,
                ] += np.eye(dim_f)
            k_z += z.T @ (face_mass_matrix_in_face @ z)
            # ------
        #     z[:, :dim_c] -= (np.linalg.inv(face_mass_matrix_in_face)) @ (hybrid_mass_matrix_in_face.T)
        #     z[:, dim_c + i * dim_f : dim_c + (i + 1) * dim_f] += np.eye(dim_f)
        #     k_z += z.T @ (face_mass_matrix_in_face @ z)
        return k_z
