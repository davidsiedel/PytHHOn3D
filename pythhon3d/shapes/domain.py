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
    def get_domain_barycenter_vector(vertices: Mat) -> Mat:
        """
        """
        shape_dimension = vertices.shape[1]
        number_of_vertices = vertices.shape[0]
        domain_barycenter = np.zeros((shape_dimension,))
        for vertex in vertices:
            domain_barycenter += vertex
        domain_barycenter = (1.0 / number_of_vertices) * domain_barycenter
        return domain_barycenter

    # @staticmethod
    # def get_domain_shape(vertices: Mat) -> str:
    #     if vertices.shape == (1,):
    #         domain_shape = "POINT"
    #     if vertices.shape == (2,):
    #         domain_shape = "SEGMENT"
    #     if vertices.shape == (2,):
    #     number_of_vertices = vertices.shape[0]
    #     problem_dimension = vertices.shape[1]
    #     if problem_dimension == 1 and number_of_vertices == 0:
    #         shape == "POINT"
    #     if problem_dimension == 1:
    #         if
    #     if number_of_vertices == 1 and problem_dimension == 1:
    #         domain_shape = "POINT"
    #     if number_of_vertices == 2 and problem_dimension == 1:
    #         domain_shape = "SEGMENT"
    #     if number_of_vertices == 3 and problem_dimension == 2:
    #         domain_shape = "TRIANGLE"
    #     if number_of_vertices == 4 and problem_dimension == 2:
    #         domain_shape = "QUADRANGLE"
    #     if number_of_vertices > 4 and problem_dimension == 2:
    #         domain_shape = "POLYGON"
    #     if number_of_vertices == 4 and problem_dimension == 3:
    #         domain_shape = "TETRAHEDRON"
    #     if number_of_vertices == 6 and problem_dimension == 3:
    #         domain_shape = "PRISM"
    #     if number_of_vertices == 8 and problem_dimension == 3:
    #         domain_shape = "HEXAHEDRON"
    #     if number_of_vertices > 8 and problem_dimension == 3:
    #         domain_shape = "POLYHEDRON"
    #     return domain_shape
