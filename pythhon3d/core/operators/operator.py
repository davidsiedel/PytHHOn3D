from core.face import Face
from core.cell import Cell
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Operator:
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
        self.cell_mass_matrix_in_cell = self.get_cell_mass_matrix_in_cell(cell, cell_basis)
        self.cell_mass_matrix_in_face = self.get_cell_mass_matrix_in_face(cell, face, cell_basis)
        self.cell_advection_matrix_in_cell = self.get_cell_advection_matrix_in_cell(
            cell, cell_basis, derivative_direction
        )
        self.hybrid_mass_matrix_in_face = self.get_hybrid_mass_matrix_in_face(cell, face, cell_basis, face_basis)

        self.cell = cell
        self.faces = faces
        self.cell_basis = cell_basis
        self.face_basis = face_basis
        self.derivative_direction = derivative_direction
        self.faces_reference_frames_transformation_matrix = faces_reference_frames_transformation_matrix

    def get_cell_mass_matrix_in_cell(self, cell: Cell, cell_basis: Basis) -> Mat:
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

    def get_cell_mass_matrix_in_face(self, cell: Cell, face: Face, cell_basis: Basis) -> Mat:
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

    def get_cell_advection_matrix_in_cell(self, cell: Cell, cell_basis: Basis, derivative_direction: int) -> Mat:
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

    def get_hybrid_mass_matrix_in_face(self, cell: Cell, face: Face, cell_basis: Basis, face_basis: Basis) -> Mat:
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
        x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, self.passmat)
        face_quadrature_nodes_in_face = Face.get_points_in_face_reference_frame(face.quadrature_nodes, self.passmat)
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
