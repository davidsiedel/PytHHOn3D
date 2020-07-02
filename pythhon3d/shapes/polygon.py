from shapes.domain import Domain
from quadratures.dunavant import DunavantRule
from shapes.triangle import Triangle
from numpy import ndarray as Mat

import numpy as np


class Polygon(Domain):
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
        if not vertices.shape[0] > 4 and not vertices.shape[1] == 2:
            raise TypeError("The domain dimension do not match that of a polygon")
        else:
            barycenter = Domain.get_domain_barycenter_vector(vertices)
            volume = 0.0
            # volume = Polygon.get_polygon_volume(vertices)
            simplicial_sub_domains = Polygon.get_polygon_simplicial_partition(vertices, barycenter)
            # sub_domains_centroids = []
            quadrature_nodes, quadrature_weights = [], []
            for simplicial_sub_domain in simplicial_sub_domains:
                # simplex_centroid = Domain.get_domain_barycenter_vector(simplicial_sub_domain)
                simplex_volume = Triangle.get_triangle_volume(simplicial_sub_domain)
                simplex_quadrature_nodes, simplex_quadrature_weights = DunavantRule.get_triangle_quadrature(
                    simplicial_sub_domain, simplex_volume, polynomial_order
                )
                volume += simplex_volume
                quadrature_nodes.append(simplex_quadrature_nodes)
                quadrature_weights.append(simplex_quadrature_weights)
                # sub_domains_centroids.append(simplex_centroid)
            # centroid = np.zeros(2,)
            # for sub_domain_centroid in sub_domains_centroids:
            #     centroid += sub_domain_centroid
            # number_of_vertices = vertices.shape[0]
            # centroid = 1.0 / number_of_vertices * centroid
            centroid = Polygon.get_polygon_centroid(vertices, volume)
            quadrature_nodes = np.concatenate(quadrature_nodes, axis=0)
            quadrature_weights = np.concatenate(quadrature_weights, axis=0)
            super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_polygon_simplicial_partition(vertices: Mat, centroid: Mat) -> Mat:
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
        number_of_vertices = vertices.shape[0]
        simplicial_sub_domains = []
        for i in range(number_of_vertices):
            sub_domain_vertices = [
                vertices[i - 1, :],
                vertices[i, :],
                centroid,
            ]
            simplicial_sub_domains.append(np.array(sub_domain_vertices))
        return simplicial_sub_domains

    @staticmethod
    def get_polygon_volume(vertices: Mat) -> float:
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
        number_of_vertices = vertices.shape[0]
        shoe_lace_matrix = []
        for i in range(number_of_vertices):
            lace = vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1]
            shoe_lace_matrix.append(lace)
        polygon_volume = np.abs(1.0 / 2.0 * np.sum(shoe_lace_matrix))
        # shoe_lace_matrix = []
        # for i in range(number_of_vertices):
        #     edge_matrix = np.array([vertices[i - 1], vertices[i]])
        #     lace = np.linalg.det(edge_matrix.T)
        #     shoe_lace_matrix.append(lace)
        # polygon_volume = np.abs(1.0 / 2.0 * np.sum(shoe_lace_matrix))
        return polygon_volume

    @staticmethod
    def get_polygon_centroid(vertices: Mat, volume: float) -> Mat:
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
        number_of_vertices = vertices.shape[0]
        # --------------------------------------------------------------------------------------------------------------
        # x coordinates
        # --------------------------------------------------------------------------------------------------------------
        centroid_matrix_x = []
        for i in range(number_of_vertices):
            centroid_matrix_x.append(
                (vertices[i - 1][0] + vertices[i][0])
                * (vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1])
            )
        polygon_centroid_x = 1.0 / (6.0 * volume) * np.sum(centroid_matrix_x)
        # --------------------------------------------------------------------------------------------------------------
        # y coordinates
        # --------------------------------------------------------------------------------------------------------------
        centroid_matrix_y = []
        for i in range(number_of_vertices):
            centroid_matrix_y.append(
                (vertices[i - 1][1] + vertices[i][1])
                * (vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1])
            )
        polygon_centroid_y = 1.0 / (6.0 * volume) * np.sum(centroid_matrix_y)
        # --------------------------------------------------------------------------------------------------------------
        # centroid
        # --------------------------------------------------------------------------------------------------------------
        polygon_centroid = np.array([polygon_centroid_x, polygon_centroid_y])
        return polygon_centroid
