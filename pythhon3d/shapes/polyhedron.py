from shapes.domain import Domain
from quadratures.dunavant import DunavantRule
from shapes.tetrahedron import Tetrahedron
from shapes.polygon import Polygon
from numpy import ndarray as Mat

import numpy as np


class Polyhedron(Domain):
    def __init__(self, vertices: Mat, connectivity_matrix: Mat):
        """
        connectivity_matrix de la forme [[0,1,2], [2,3,6], [2,8,5],...] avec la liste des noeuds en tableau organisés par face dans le repère local := declaration de l'element
        """
        if not vertices.shape[0] > 4 and not vertices.shape[1] == 2:
            raise TypeError("The domain dimensions do not match that of a polygon")
        else:
            barycenter = Domain.get_domain_barycenter_vector(vertices_matrix)
            simplicial_sub_domains = Polyhedron.get_polyhedron_simplicial_partition(
                vertices, connectivity_matrix, barycenter
            )
            volume = 0.0
            quadrature_nodes, quadrature_weights = [], []
            for simplicial_sub_domain in simplicial_sub_domains:
                simplex_volume = Tetrahedron.get_tetrahedron_volume(simplicial_sub_domain)
                simplex_quadrature_nodes, simplex_quadrature_weights = DunavantRule.get_tetrahedron_quadrature(
                    simplicial_sub_domain, simplex_volume
                )
                volume += simplex_volume
                quadrature_nodes.append(simplex_quadrature_nodes)
                quadrature_weights.append(simplex_quadrature_weights)
            quadrature_nodes = np.concatenate(quadrature_nodes, axis=0)
            quadrature_weights = np.concatenate(quadrature_weights, axis=0)
            super().__init__(barycenter, volume, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_polyhedron_simplicial_partition(vertices: Mat, connectivity_matrix: Mat, barycenter: Mat) -> Mat:
        """
        connectivity_matrix de la forme [[0,1,2], [2,3,6], [2,8,5],...] avec la liste des noeuds en tableau organisés par face dans le repère local := declaration de l'element
        """
        tetras = []
        for face_vertices_indexes in connectivity_matrix:
            face_vertices = vertices[face_vertices_indexes]
            if face_vertices.shape[0] > 3:
                face_barycenter = Domain.get_domain_barycenter_vector(face_vertices)
                sub_faces = Polygon.get_polygon_simplicial_partition(face_vertices, face_barycenter)
                for sub_face in sub_faces:
                    tetra = np.concatenate((sub_face, barycenter), axis=0)
                    tetras.append(tetras)
            else:
                tetra = np.concatenate((face_vertices, barycenter), axis=0)
                tetras.append(tetras)
        return tetras
