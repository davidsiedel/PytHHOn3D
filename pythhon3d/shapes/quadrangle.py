from shapes.domain import Domain
from shapes.triangle import Triangle
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat

import numpy as np


class Quadrangle(Domain):
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
        if not vertices.shape == (3, 2):
            raise TypeError("The domain vertices do not match that of a triangle")
        else:
            centroid = Domain.get_domain_barycenter_vector(vertices)
            volume = Quadrangle.get_quadrangle_volume(vertices)
            diameter = Quadrangle.get_quadrangle_diameter(vertices)
            quadrature_nodes, quadrature_weights = DunavantRule.get_quadrangle_quadrature(
                vertices, volume, polynomial_order
            )
            super().__init__(centroid, volume, diameter, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_quadrangle_volume(vertices: Mat) -> float:
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
        t0 = vertices[0, 1, 2]
        t0 = vertices[0, 3, 2]
        v0 = Triangle.get_triangle_volume(t0)
        v1 = Triangle.get_triangle_volume(t1)
        return v0 + v1

    @staticmethod
    def get_quadrangle_diameter(vertices: Mat) -> float:
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
        shape_dimension = 2
        diag0 = vertices[2] - vertices[0]
        diag1 = vertices[3] - vertices[1]
        quadrangle_diameter = max([np.sqrt((diag[1] + diag[0]) ** 2) for diag in [diag0, diag1]])
        return quadrangle_diameter
