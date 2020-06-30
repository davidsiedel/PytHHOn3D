from shapes.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat


class Point(Domain):
    def __init__(self, vertices: Mat):
        """
        """
        if not vertices.shape == (1, 1):
            raise TypeError("The domain vertices do not match that of a point")
        else:
            barycenter = vertices[0]
            volume = 1.0
            quadrature_nodes, quadrature_weights = DunavantRule.get_point_quadrature()
            super().__init__(barycenter, volume, quadrature_nodes, quadrature_weights)
