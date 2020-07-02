from shapes.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat

import numpy as np


class Segment(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
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
        if not vertices.shape == (2, 1):
            raise TypeError("The domain vertices do not match that of a line")
        else:
            centroid = Domain.get_domain_barycenter_vector(vertices)
            volume = Segment.get_segment_volume(vertices)
            quadrature_nodes, quadrature_weights = DunavantRule.get_segment_quadrature(
                vertices, volume, polynomial_order
            )
            super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_segment_volume(vertices: Mat) -> float:
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
        segment_volume = np.abs(vertices[0, 0] - vertices[1, 0])
        return segment_volume
