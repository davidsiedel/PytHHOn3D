from shapes.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat

import numpy as np


class Triangle(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
        """
        """
        if not vertices.shape == (3, 2):
            raise TypeError("The domain vertices do not match that of a triangle")
        else:
            centroid = Domain.get_domain_barycenter_vector(vertices)
            volume = Triangle.get_triangle_volume(vertices)
            quadrature_nodes, quadrature_weights = DunavantRule.get_triangle_quadrature(
                vertices, volume, polynomial_order
            )
            super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_triangle_volume(vertices: Mat) -> float:
        """
        """
        shape_dimension = 2
        triangle_origin_vertex_vector = np.tile(vertices[0], (shape_dimension + 1, 1))
        triangle_edges = (vertices - triangle_origin_vertex_vector)[1:]
        triangle_volume = np.abs(1.0 / 2.0 * np.linalg.det(triangle_edges))
        return triangle_volume
