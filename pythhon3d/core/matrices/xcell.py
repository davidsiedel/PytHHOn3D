import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from shapes.domain import Domain
from core.cell import Cell
from core.unknown import Unknown
from core.integration import Integration
from bases.basis import Basis

from shapes.segment import Segment
from shapes.triangle import Triangle
from shapes.polygon import Polygon
from shapes.tetrahedron import Tetrahedron
from shapes.polyhedron import Polyhedron


class Xcell:
    def __init__(self, cell: Cell, cell_basis_l: Basis, cell_basis_k: Basis, cell_basis_k1: Basis, unknown: Unknown):
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
        # Building the cell
        # --------------------------------------------------------------------------------------------------------------
        centroid = cell.centroid
        volume = cell.volume
        quadrature_points = cell.quadrature_points
        quadrature_weights = cell.quadrature_weights
        # --------------------------------------------------------------------------------------------------------------
        super().__init__(centroid, volume, diameter, quadrature_points, quadrature_weights)
        m_phi_k_phi_k = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_k, cell_basis_k)
        m_phi_k_phi_k_inv = np.linalg.inv(m_phi_k_phi_k)
        m_grad_phi_k1_grad_phi_k1_list = []
        m_phi_k_grad_phi_l_list = []
        m_grad_phi_k1_phi_l_list = []
        for j in range(unknown.problem_dimension):
            # ----------------------------------------------------------------------------------------------------------
            m_grad_phi_k1_grad_phi_k1_j = Integration.get_cell_stiffness_matrix_in_cell(
                cell, cell_basis_k1, cell_basis_k1, j
            )
            m_grad_phi_k1_grad_phi_k1_list.append(m_grad_phi_k1_grad_phi_k1_j)
            # ----------------------------------------------------------------------------------------------------------
            m_phi_k_grad_phi_l_j = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_k, cell_basis_l, j)
            m_phi_k_grad_phi_l_list.append(m_phi_k_grad_phi_l_j)
            # ----------------------------------------------------------------------------------------------------------
            m_phi_l_grad_phi_k1_j = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_l, cell_basis_k1, j)
            m_grad_phi_k1_phi_l_j = m_phi_l_grad_phi_k1_j.T
            m_grad_phi_k1_phi_l_list.append(m_grad_phi_k1_phi_l_j)
