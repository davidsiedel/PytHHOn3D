from core.face import Face
from core.cell import Cell
from core.operators import Operators
from core.elements.element import Element
from core.elements.hdg import HDG
from behaviors.behavior import Behavior

# from core.operators.operator import Operator
# from core.operators.gradient import Gradient
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Elasticity(Behavior):
    def __init__(
        self, problem_dimension: int, field_dimension: int, local_gradients: List[Mat], stabilization_stifness: float
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
        # --------------------------------------------------------------------------------------------------------------
        # Checking the consistency result
        # --------------------------------------------------------------------------------------------------------------
        self.check_field_dimension_consistency(problem_dimension, field_dimension)
        lame_lambda = 1.0
        lame_mu = 2.0
        if field_dimension == 2 and problem_dimension == 2:
            for (i, j) in [(0, 0), (1, 1), (2, 2), (2, 3), (1, 3), (1, 2)]:
                pass
        # problem_dimension, field_dimension, mechanical_operator, stabilization_stifness, stabilization_operator
        super().__init__(
            problem_dimension, field_dimension, mechanical_operator, stabilization_stifness, stabilization_operator
        )

    def get_tangent_matrix():
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
        if field_dimension == 2 and problem_dimension == 2:
            lame_lambda = 1.0
            lame_mu = 2.0
            np.array([[]])


# for (i, j) in [(0, 0), (1, 1), (2, 2), (2, 3), (1, 3), (1, 2)]:
#     pass

