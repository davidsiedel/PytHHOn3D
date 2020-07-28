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


class Condensation:
    def __init__(self):
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

    @staticmethod
    def get_system_decomposition(matrix: Mat, vector: Mat, unknown: Unknown, cell_basis: Basis) -> Mat:
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
        l1 = unknown.field_dimension * cell_basis.basis_dimension
        m_cell_cell = matrix[:l1, :l1]
        m_cell_faces = matrix[:l1, l1:]
        m_faces_cell = matrix[l1:, :l1]
        m_faces_faces = matrix[l1:, l1:]
        v_cell = vector[:l1]
        v_faces = vector[l1:]
        # --------------------------------------------------------------------------------------------------------------
        m_cell_cell_inv = np.linalg.inv(m_cell_cell)
        # --------------------------------------------------------------------------------------------------------------
        return m_cell_cell_inv, m_cell_faces, m_faces_cell, m_faces_faces, v_cell, v_faces

    @staticmethod
    def get_condensated_system(
        m_cell_cell_inv: Mat, m_cell_faces: Mat, m_faces_cell: Mat, m_faces_faces: Mat, v_cell: Mat, v_faces: Mat
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
        ge = m_faces_cell @ m_cell_cell_inv
        gd = ge @ m_cell_faces
        # --------------------------------------------------------------------------------------------------------------
        # mat
        # --------------------------------------------------------------------------------------------------------------
        m_cond = m_faces_faces - gd
        # --------------------------------------------------------------------------------------------------------------
        # vec
        # --------------------------------------------------------------------------------------------------------------
        v_cond = v_faces - ge @ v_cell
        return m_cond, v_cond

    @staticmethod
    def get_cell_unknown(m_cell_cell_inv: Mat, m_cell_faces: Mat, v_cell: Mat, x_faces: Mat) -> Mat:
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
        x_cell = m_cell_cell_inv @ v_cell - m_cell_cell_inv @ m_cell_faces @ x_faces
        return x_cell
