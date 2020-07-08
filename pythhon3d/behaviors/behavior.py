from core.face import Face
from core.cell import Cell
from core.operators import Operators
from core.elements.element import Element
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Behavior:
    def __init__(
        self, problem_dimension: int, field_dimension: int, tangent_matrix: Mat,
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
        self.tangent_matrix = tangent_matrix

    def check_field_dimension_consistency(self, problem_dimension: int, field_dimension: int) -> None:
        if field_dimension > problem_dimension:
            raise ValueError("too high")
        else:
            return

