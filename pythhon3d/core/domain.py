import numpy as np
from numpy import ndarray as Mat
from core.quadrature import Quadrature


class Domain:
    def __init__(self, domain_vertices_matrix: Mat):
        """
        """
        self.barycenter_vector = self.get_domain_barycenter_vector(domain_vertices_matrix)

    def get_domain_barycenter_vector(self, domain_vertices_matrix: Mat) -> Mat:
        """
        """
        number_of_vertices = domain_vertices_matrix.shape[0]
        d = domain_vertices_matrix.shape[1]
        domain_barycenter = np.zeros((d,))
        for vertex in domain_vertices_matrix:
            domain_barycenter += vertex
        domain_barycenter = (1.0 / number_of_vertices) * domain_barycenter
        return domain_barycenter.reshape(1, d)

    def get_simplex_volume(self, simplex_vertices_matrix):
        """
        """
        d = simplex_vertices_matrix.shape[1]
        number_of_vertices = simplex_vertices_matrix.shape[0]
        if number_of_vertices > 1:
            simplex_origin_vertex_vector = np.tile(simplex_vertices_matrix[0], (d + 1, 1))
            simplex_edges = (simplex_vertices_matrix - simplex_origin_vertex_vector)[1:]
            if d == 1:
                simplex_volume = np.abs(1.0 / 1.0 * np.linalg.det(simplex_edges))
            if d == 2:
                simplex_volume = np.abs(1.0 / 2.0 * np.linalg.det(simplex_edges))
            if d == 3:
                simplex_volume = np.abs(1.0 / 6.0 * np.linalg.det(simplex_edges))
        elif number_of_vertices == 1 and d == 1:
            simplex_volume = 1.0
        return simplex_volume

    def get_simplex_quadrature(self, simplex_vertices_matrix, simplex_volume, k):
        """
        """
        d = simplex_vertices_matrix.shape[1]
        number_of_vertices = simplex_vertices_matrix.shape[0]
        if number_of_vertices > 1:
            if d == 1:
                quadrature_points, quadrature_weights = Quadrature.get_unite_segment_quadrature(
                    simplex_vertices_matrix, simplex_volume, k
                )
                jacobian_value = 1.0 / simplex_volume
            if d == 2:
                quadrature_points, quadrature_weights = Quadrature.get_unite_triangle_quadrature(
                    simplex_vertices_matrix, simplex_volume, k
                )
                jacobian_value = (1.0 / 2.0) / simplex_volume
            if d == 3:
                quadrature_points, quadrature_weights = Quadrature.get_unite_tetrahedron_quadrature(
                    simplex_vertices_matrix, simplex_volume, k
                )
                jacobian_value = (1.0 / 6.0) / simplex_volume
        elif number_of_vertices == 1 and d == 1:
            quadrature_points, quadrature_weights = Quadrature.get_unite_point_quadrature()
            jacobian_value = 1.0
        quadrature_weights = quadrature_weights * jacobian_value
        return quadrature_points, quadrature_weights
