from core.face import Face
from core.cell import Cell
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Integration:
    def __init__(self):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Integration class provides methods to compute integration matrices that are then used to define elements.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        All methods are static, so no parameters are needed.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        All methods are static, so no attributes are created.
        """

    @staticmethod
    def get_cell_mass_matrix_in_cell(cell: Cell, cell_basis: Basis) -> Mat:
        """
        ================================================================================================================
        Description :
        ================================================================================================================
        Returns the mass matrix in a cell-like domain using quadrature points and weights
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - cell : the considered cell
        - cell_basis : the polynomial basis of the cell
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let linear polynomials in a one dimensional cell C. The mass matrix is then the sum for all quadrature points
        x_Q and quadrature weights w_Q of the following matricial contribution:
        w_Q*| 1*1    x_Q*1  |
            | 1*x_Q x_Q*x_Q |
        """
        v_c = cell.volume
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        cell_mass_matrix_in_cell = np.zeros((cell_basis.basis_dimension, cell_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_c, w_Q_c in zip(cell.quadrature_nodes, cell.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector = cell_basis.get_phi_vector(x_Q_c, x_c, v_c)
            number_of_components = phi_vector.shape[0]
            phi_vector = np.resize(phi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_c * phi_vector.T @ phi_vector
            cell_mass_matrix_in_cell += m
        return cell_mass_matrix_in_cell

    @staticmethod
    def get_cell_stiffness_matrix_in_cell(cell: Cell, cell_basis: Basis) -> Mat:
        """
        ================================================================================================================
        Description :
        ================================================================================================================
        Returns the mass matrix in a cell-like domain using quadrature points and weights
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - cell : the considered cell
        - cell_basis : the polynomial basis of the cell
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let linear polynomials in a one dimensional cell C. The mass matrix is then the sum for all quadrature points
        x_Q and quadrature weights w_Q of the following matricial contribution:
        w_Q*| 0*0 1*0 |
            | 0*1 1*1 |
        """
        v_c = cell.volume
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        cell_stifness_matrix_in_cell = np.zeros((cell_basis.basis_dimension, cell_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_c, w_Q_c in zip(cell.quadrature_nodes, cell.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            d_phi_vector = cell_basis.get_d_phi_vector(x_Q_c, x_c, v_c)
            number_of_components = d_phi_vector.shape[0]
            d_phi_vector = np.resize(d_phi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_c * d_phi_vector.T @ d_phi_vector
            cell_stifness_matrix_in_cell += m
        return cell_stifness_matrix_in_cell

    @staticmethod
    def get_cell_mass_matrix_in_face(cell: Cell, face: Face, cell_basis: Basis) -> Mat:
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
        v_f = face.volume
        x_f = face.centroid
        # --------------------------------------------------------------------------------------------------------------
        cell_mass_matrix_in_face = np.zeros((cell_basis.basis_dimension, cell_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_f, w_Q_f in zip(face.quadrature_nodes, face.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector = cell_basis.get_phi_vector(x_Q_f, x_f, v_f)
            number_of_components = phi_vector.shape[0]
            phi_vector = np.resize(phi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_f * phi_vector.T @ phi_vector
            cell_mass_matrix_in_face += m
        return cell_mass_matrix_in_face

    @staticmethod
    def get_cell_advection_matrix_in_cell(cell: Cell, cell_basis: Basis, derivative_direction: int) -> Mat:
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
        v_c = cell.volume
        x_c = cell.centroid
        dx = derivative_direction
        # --------------------------------------------------------------------------------------------------------------
        cell_advection_matrix_in_cell = np.zeros((cell_basis.basis_dimension, cell_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_c, w_Q_c in zip(cell.quadrature_nodes, cell.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector = cell_basis.get_phi_vector(x_Q_c, x_c, v_c)
            number_of_components = phi_vector.shape[0]
            phi_vector = np.resize(phi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            d_phi_vector = cell_basis.get_d_phi_vector(x_Q_c, x_c, v_c, dx)
            number_of_components = d_phi_vector.shape[0]
            d_phi_vector = np.resize(phi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_c * phi_vector.T @ d_phi_vector
            cell_advection_matrix_in_cell += m
        return cell_advection_matrix_in_cell

    @staticmethod
    def get_hybrid_mass_matrix_in_face(
        cell: Cell, face: Face, cell_basis: Basis, face_basis: Basis, face_reference_frame_transformation_matrix: Mat,
    ) -> Mat:
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
        v_f = face.volume
        x_f = face.centroid
        # --------------------------------------------------------------------------------------------------------------
        hybrid_mass_matrix_in_face = np.zeros((cell_basis.basis_dimension, face_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, face_reference_frame_transformation_matrix)
        face_quadrature_nodes_in_face = Face.get_points_in_face_reference_frame(
            face.quadrature_nodes, face_reference_frame_transformation_matrix
        )
        # --------------------------------------------------------------------------------------------------------------
        for x_Q, x_Q_in_face, w_Q in zip(face.quadrature_nodes, face_quadrature_nodes_in_face, face.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector = cell_basis.get_phi_vector(x_Q, x_f, v_f)
            number_of_components = phi_vector.shape[0]
            phi_vector = np.resize(phi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            psi_vector = face_basis.get_phi_vector(x_Q_in_face, x_f_in_face, v_f)
            number_of_components = psi_vector.shape[0]
            psi_vector = np.resize(psi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q * phi_vector.T @ psi_vector
            hybrid_mass_matrix_in_face += m
        return hybrid_mass_matrix_in_face

    @staticmethod
    def get_face_mass_matrix_in_face(
        face: Face, face_basis: Basis, face_reference_frame_transformation_matrix: Mat
    ) -> Mat:
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
        v_f = face.volume
        x_f = face.centroid
        # --------------------------------------------------------------------------------------------------------------
        face_mass_matrix_in_face = np.zeros((face_basis.basis_dimension, face_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, face_reference_frame_transformation_matrix)
        face_quadrature_nodes_in_face = Face.get_points_in_face_reference_frame(
            face.quadrature_nodes, face_reference_frame_transformation_matrix
        )
        for x_Q_f_in_face, w_Q_f in zip(face_quadrature_nodes_in_face, face.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            psi_vector = face_basis.get_phi_vector(x_Q_f_in_face, x_f_in_face, v_f)
            number_of_components = psi_vector.shape[0]
            psi_vector = np.resize(psi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_f * psi_vector.T @ psi_vector
            face_mass_matrix_in_face += m
        return face_mass_matrix_in_face
