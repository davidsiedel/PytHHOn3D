import numpy as np
from typing import List
from numpy import ndarray as Mat

from core.face import Face
from core.cell import Cell


class Unknown:
    def __init__(self, problem_dimension: int, field_dimension: int, symmetric_gradient: bool = False):
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
        if problem_dimension == field_dimension:
            if not symmetric_gradient:
                if field_dimension == 1:
                    indices = [(0, 0)]
                elif field_dimension == 2:
                    indices = [(0, 0), (1, 1), (0, 1), (1, 0)]
                elif field_dimension == 3:
                    indices = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1), (2, 1), (2, 0), (1, 0)]
                else:
                    raise ValueError("the field_dimension is not correct")
            else:
                if field_dimension == 1:
                    indices = [(0, 0)]
                elif field_dimension == 2:
                    indices = [(0, 0), (1, 1), (0, 1)]
                elif field_dimension == 3:
                    indices = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
                else:
                    raise ValueError("the field_dimension is not correct")
        elif problem_dimension > field_dimension:
            if field_dimension == 1:
                indices = [(0, 0)]
            elif field_dimension == 2:
                indices = [(0, 0), (0, 1)]
            elif field_dimension == 3:
                indices = [(0, 0), (0, 1), (0, 2)]
            else:
                raise ValueError("the field_dimension is not correct")
        # --------------------------------------------------------------------------------------------------------------
        self.problem_dimension = problem_dimension
        self.field_dimension = field_dimension
        self.indices = indices
        self.symmetric_gradient = symmetric_gradient
        # self.symmetric_gradient = symmetric_gradient
        # self.field_dimension = field_dimension
