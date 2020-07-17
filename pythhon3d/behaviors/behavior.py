import numpy as np
from typing import List
from numpy import ndarray as Mat

from operators.operator import Operator
from behaviors.behavior import Behavior


class Behavior:
    def __init__(
        self, problem_dimension: int, field_dimension: int, indices: List[tuple], tangent_matrix: Mat,
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
        self.problem_dimension = problem_dimension
        self.field_dimension = field_dimension
        self.indices = indices
        self.tangent_matrix = tangent_matrix
