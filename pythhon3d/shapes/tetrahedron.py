from shapes.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat

import numpy as np


class Tetrahedron(Domain):
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
        if not vertices.shape == (4, 3):
            raise TypeError("The domain vertices do not match that of a tetrahedron")
        else:
            barycenter = Domain.get_domain_barycenter_vector(vertices)
            volume = Tetrahedron.get_tetrahedron_volume(vertices)
            quadrature_nodes, quadrature_weights = DunavantRule.get_tetrahedron_quadrature(
                vertices, volume, polynomial_order
            )
            super().__init__(barycenter, volume, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_tetrahedron_volume(vertices: Mat) -> float:
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
        shape_dimension = 3
        tetrahedron_origin_vertex_vector = np.tile(vertices[0], (shape_dimension + 1, 1))
        tetrahedron_edges = (vertices - tetrahedron_origin_vertex_vector)[1:]
        tetrahedron_volume = np.abs(1.0 / 6.0 * np.linalg.det(tetrahedron_edges))
        return tetrahedron_volume
