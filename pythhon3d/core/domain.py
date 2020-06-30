import numpy as np
from numpy import ndarray as Mat
from core.quadrature import Quadrature
from enum import Enum


class Domain:
    def __init__(self, shape: str, domain_vertices_matrix: Mat):
        """
        """
        self.barycenter_vector = self.get_domain_barycenter_vector(domain_vertices_matrix)

    def get_domain_barycenter_vector(self, shape: str, domain_vertices_matrix: Mat) -> Mat:
        """
        """
        if shape == "POINT":
            domain_barycenter = domain_vertices_matrix[0]
        if shape == "LINE":
            d = 1
            number_of_vertices = 2
            domain_barycenter = np.zeros((d,))
            for vertex in domain_vertices_matrix:
                domain_barycenter += vertex
            domain_barycenter = (1.0 / number_of_vertices) * domain_barycenter
        if shape == "SURFACE":
            d = 2
            number_of_vertices = domain_vertices_matrix.shape[0]
            domain_barycenter = np.zeros((d,))
            for vertex in domain_vertices_matrix:
                domain_barycenter += vertex
            domain_barycenter = (1.0 / number_of_vertices) * domain_barycenter
        if shape == "VOLUME":
            d = 3
            number_of_vertices = domain_vertices_matrix.shape[0]
            domain_barycenter = np.zeros((d,))
            for vertex in domain_vertices_matrix:
                domain_barycenter += vertex
            domain_barycenter = (1.0 / number_of_vertices) * domain_barycenter
        return domain_barycenter

    def get_simplex_volume(self, shape: str, simplex_vertices_matrix):
        """
        """
        if shape == "POINT":
            simplex_volume = 1.0
        if shape == "LINE":
            simplex_origin_vertex_vector = np.tile(simplex_vertices_matrix[0], (d + 1, 1))
            simplex_edges = (simplex_vertices_matrix - simplex_origin_vertex_vector)[1:]
            simplex_volume = np.abs(1.0 / 1.0 * np.linalg.det(simplex_edges))
        if shape == "SURFACE":
            simplex_origin_vertex_vector = np.tile(simplex_vertices_matrix[0], (d + 1, 1))
            simplex_edges = (simplex_vertices_matrix - simplex_origin_vertex_vector)[1:]
            simplex_volume = np.abs(1.0 / 2.0 * np.linalg.det(simplex_edges))
        if shape == "VOLUME":
            simplex_origin_vertex_vector = np.tile(simplex_vertices_matrix[0], (d + 1, 1))
            simplex_edges = (simplex_vertices_matrix - simplex_origin_vertex_vector)[1:]
            simplex_volume = np.abs(1.0 / 6.0 * np.linalg.det(simplex_edges))
        return simplex_volume

    def get_simplex_quadrature(self, simplex_vertices_matrix, simplex_volume, k):
        """
        """
        if shape == "POINT":
            quadrature_points, quadrature_weights = Quadrature.get_unite_point_quadrature()
            jacobian_value = 1.0
        if shape == "LINE":
            quadrature_points, quadrature_weights = Quadrature.get_unite_segment_quadrature(
                simplex_vertices_matrix, simplex_volume, k
            )
            jacobian_value = 1.0 / simplex_volume
        if shape == "SURFACE":
            quadrature_points, quadrature_weights = Quadrature.get_unite_triangle_quadrature(
                simplex_vertices_matrix, simplex_volume, k
            )
            jacobian_value = (1.0 / 2.0) / simplex_volume
        if shape == "VOLUME":
            quadrature_points, quadrature_weights = Quadrature.get_unite_tetrahedron_quadrature(
                simplex_vertices_matrix, simplex_volume, k
            )
            jacobian_value = (1.0 / 6.0) / simplex_volume
        quadrature_weights = quadrature_weights * jacobian_value
        return quadrature_points, quadrature_weights
