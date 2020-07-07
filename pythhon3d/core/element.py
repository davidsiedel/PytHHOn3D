from core.face import Face
from core.cell import Cell
from core.operators.operator import Operator
from core.operators.gradient import Gradient
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Element:
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
        passmats = []
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
            passmats.append(passmat)
        # --------------------------------------------------------------------------------------------------------------
        reconstructed_gradient = Gradient(
            cell, faces, cell_basis, face_basis, derivative_direction, passmats
        ).reconstructed_gradient
        print(reconstructed_gradient)
        # --------------------------------------------------------------------------------------------------------------
        #     # ----------------------------------------------------------------------------------------------------------
        #     # Computing the reconstructed gradient operator matrix
        #     # ----------------------------------------------------------------------------------------------------------
        #     m_cell_mass_left = self.get_cell_mass_matrix_in_cell(cell, cell_basis)
        #     m_cell_advc_right = self.get_cell_advection_matrix_in_cell(cell, cell_basis, derivative_direction)
        #     m_cell_mass_right = self.get_cell_mass_matrix_in_face(cell, face, cell_basis)
        #     m_hybr_mass_right = self.get_hybrid_mass_matrix_in_face(cell, face, cell_basis, face_basis)
        #     # ----------------------------------------------------------------------------------------------------------
        #     b_cell = m_cell_advc_right - m_cell_mass_right
        #     # ----------------------------------------------------------------------------------------------------------
        #     face_normal_vector = self.passmat[-1]
        #     b_face = m_hybr_mass_right * face_normal_vector[direction]
        #     b_faces.append(b_face)
        # # --------------------------------------------------------------------------------------------------------------
        # b_faces = np.concatenate(b_faces, axis=1)
        # b_left = np.concatenate((b_cell, b_faces), axis=1)
        # print("m_cell_mass_left : \n{}".format(m_cell_mass_left))
        # reconstructed_gradient_operator = np.linalg.inv(m_cell_mass_left) @ b_left
        # # --------------------------------------------------------------------------------------------------------------
        # # Computing the stabilization operator matrix
        # # --------------------------------------------------------------------------------------------------------------
        # print("reconstructed_gradient_operator : \n{}".format(reconstructed_gradient_operator))

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

    # def get_cell_mass_matrix_in_cell(self, cell: Cell, cell_basis: Basis) -> Mat:
    #     """
    #     ================================================================================================================
    #     Description :
    #     ================================================================================================================

    #     ================================================================================================================
    #     Parameters :
    #     ================================================================================================================

    #     ================================================================================================================
    #     Exemple :
    #     ================================================================================================================

    #     """
    #     v_c = cell.volume
    #     x_c = cell.centroid
    #     # --------------------------------------------------------------------------------------------------------------
    #     cell_mass_matrix_in_cell = np.zeros((cell_basis.basis_dimension, cell_basis.basis_dimension))
    #     # --------------------------------------------------------------------------------------------------------------
    #     for x_Q_c, w_Q_c in zip(cell.quadrature_nodes, cell.quadrature_weights):
    #         # ----------------------------------------------------------------------------------------------------------
    #         phi_vector = cell_basis.get_phi_vector(x_Q_c, x_c, v_c)
    #         number_of_components = phi_vector.shape[0]
    #         phi_vector = np.resize(phi_vector, (1, number_of_components))
    #         # ----------------------------------------------------------------------------------------------------------
    #         m = w_Q_c * phi_vector.T @ phi_vector
    #         cell_mass_matrix_in_cell += m
    #     return cell_mass_matrix_in_cell

    # def get_cell_mass_matrix_in_face(self, cell: Cell, face: Face, cell_basis: Basis) -> Mat:
    #     """
    #     ================================================================================================================
    #     Description :
    #     ================================================================================================================

    #     ================================================================================================================
    #     Parameters :
    #     ================================================================================================================

    #     ================================================================================================================
    #     Exemple :
    #     ================================================================================================================

    #     """
    #     v_f = face.volume
    #     x_f = face.centroid
    #     # --------------------------------------------------------------------------------------------------------------
    #     cell_mass_matrix_in_face = np.zeros((cell_basis.basis_dimension, cell_basis.basis_dimension))
    #     # --------------------------------------------------------------------------------------------------------------
    #     for x_Q_f, w_Q_f in zip(face.quadrature_nodes, face.quadrature_weights):
    #         # ----------------------------------------------------------------------------------------------------------
    #         phi_vector = cell_basis.get_phi_vector(x_Q_f, x_f, v_f)
    #         number_of_components = phi_vector.shape[0]
    #         phi_vector = np.resize(phi_vector, (1, number_of_components))
    #         # ----------------------------------------------------------------------------------------------------------
    #         m = w_Q_f * phi_vector.T @ phi_vector
    #         cell_mass_matrix_in_face += m
    #     return cell_mass_matrix_in_face

    # def get_cell_advection_matrix_in_cell(self, cell: Cell, cell_basis: Basis, derivative_direction: int) -> Mat:
    #     """
    #     ================================================================================================================
    #     Description :
    #     ================================================================================================================

    #     ================================================================================================================
    #     Parameters :
    #     ================================================================================================================

    #     ================================================================================================================
    #     Exemple :
    #     ================================================================================================================

    #     """
    #     v_c = cell.volume
    #     x_c = cell.centroid
    #     dx = derivative_direction
    #     # --------------------------------------------------------------------------------------------------------------
    #     cell_advection_matrix_in_cell = np.zeros((cell_basis.basis_dimension, cell_basis.basis_dimension))
    #     # --------------------------------------------------------------------------------------------------------------
    #     for x_Q_c, w_Q_c in zip(cell.quadrature_nodes, cell.quadrature_weights):
    #         # ----------------------------------------------------------------------------------------------------------
    #         phi_vector = cell_basis.get_phi_vector(x_Q_c, x_c, v_c)
    #         number_of_components = phi_vector.shape[0]
    #         phi_vector = np.resize(phi_vector, (1, number_of_components))
    #         # ----------------------------------------------------------------------------------------------------------
    #         d_phi_vector = cell_basis.get_d_phi_vector(x_Q_c, x_c, v_c, dx)
    #         number_of_components = d_phi_vector.shape[0]
    #         d_phi_vector = np.resize(phi_vector, (1, number_of_components))
    #         # ----------------------------------------------------------------------------------------------------------
    #         m = w_Q_c * phi_vector.T @ d_phi_vector
    #         cell_advection_matrix_in_cell += m
    #     return cell_advection_matrix_in_cell

    # def get_hybrid_mass_matrix_in_face(self, cell: Cell, face: Face, cell_basis: Basis, face_basis: Basis) -> Mat:
    #     """
    #     ================================================================================================================
    #     Description :
    #     ================================================================================================================

    #     ================================================================================================================
    #     Parameters :
    #     ================================================================================================================

    #     ================================================================================================================
    #     Exemple :
    #     ================================================================================================================

    #     """
    #     v_f = face.volume
    #     x_f = face.centroid
    #     # --------------------------------------------------------------------------------------------------------------
    #     hybrid_mass_matrix_in_face = np.zeros((cell_basis.basis_dimension, face_basis.basis_dimension))
    #     # --------------------------------------------------------------------------------------------------------------
    #     x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, self.passmat)
    #     face_quadrature_nodes_in_face = Face.get_points_in_face_reference_frame(face.quadrature_nodes, self.passmat)
    #     # --------------------------------------------------------------------------------------------------------------
    #     for x_Q, x_Q_in_face, w_Q in zip(face.quadrature_nodes, face_quadrature_nodes_in_face, face.quadrature_weights):
    #         # ----------------------------------------------------------------------------------------------------------
    #         phi_vector = cell_basis.get_phi_vector(x_Q, x_f, v_f)
    #         number_of_components = phi_vector.shape[0]
    #         phi_vector = np.resize(phi_vector, (1, number_of_components))
    #         # ----------------------------------------------------------------------------------------------------------
    #         psi_vector = face_basis.get_phi_vector(x_Q_in_face, x_f_in_face, v_f)
    #         number_of_components = psi_vector.shape[0]
    #         psi_vector = np.resize(psi_vector, (1, number_of_components))
    #         # ----------------------------------------------------------------------------------------------------------
    #         m = w_Q * phi_vector.T @ psi_vector
    #         hybrid_mass_matrix_in_face += m
    #     return hybrid_mass_matrix_in_face
