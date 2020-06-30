from typing import List
from numpy import ndarray as Mat
import numpy as np


class Domain:
    def __init__(self, centroid: Mat, volume: float, quadrature_nodes: List[Mat], quadrature_weights: List[Mat]):
        """
        """
        self.centroid = centroid
        self.volume = volume
        self.quadrature_nodes = quadrature_nodes
        self.quadrature_weights = quadrature_weights

    @staticmethod
    def get_domain_barycenter_vector(vertices_matrix: Mat) -> Mat:
        """
        """
        shape_dimension = vertices_matrix.shape[1]
        number_of_vertices = vertices_matrix.shape[0]
        domain_barycenter = np.zeros((shape_dimension,))
        for vertex in vertices_matrix:
            domain_barycenter += vertex
        domain_barycenter = (1.0 / number_of_vertices) * domain_barycenter
        return domain_barycenter
