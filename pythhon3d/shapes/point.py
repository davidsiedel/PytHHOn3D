from shapes.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat
import numpy as np


class Point(Domain):
    def __init__(self, vertices: Mat):
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
        if not vertices.shape == (1, 1):
            raise TypeError("The domain vertices do not match that of a point")
        else:
            # centroid = vertices[0]
            centroid = np.array([])
            volume = 1.0
            quadrature_nodes, quadrature_weights = DunavantRule.get_point_quadrature()
            super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)