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


class Element:
    def __init__(
        self,
        # vertices: Mat,
        cell: Cell,
        # faces: List[Face],
        local_gradient_operator: Mat,
        local_stabilization_form: Mat,
        local_mass_operator: Mat,
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
        # self.vertices = vertices
        self.cell = cell
        # self.faces = faces
        self.local_gradient_operator = local_gradient_operator
        self.local_mass_operator = local_mass_operator
        self.local_stabilization_form = local_stabilization_form
