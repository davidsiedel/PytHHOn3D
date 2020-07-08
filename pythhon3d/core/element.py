import numpy as np
from typing import List
from numpy import ndarray as Mat

from operators.operator import Operator
from behaviors.behavior import Behavior


class Element:
    def __init__(
        self,
        vertices: Mat,
        quadrature_points: Mat,
        behavior: Behavior,
        local_mechanical_mass_operator: Mat,
        local_mechanical_stiffness_operators: List[Mat],
        local_stabilization_stiffness_operator: Mat,
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
