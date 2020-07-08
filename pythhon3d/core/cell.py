import numpy as np
from numpy import ndarray as Mat

from shapes.domain import Domain

from shapes.segment import Segment
from shapes.triangle import Triangle
from shapes.polygon import Polygon
from shapes.tetrahedron import Tetrahedron
from shapes.polyhedron import Polyhedron


class Cell(Domain):
    def __init__(self, vertices: Mat, connectivity_matrix: Mat, polynomial_order: int):
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
        cell_shape = Cell.get_cell_shape(vertices)
        if cell_shape == "SEGMENT":
            c = Segment(vertices, polynomial_order)
            centroid = c.centroid
            volume = c.volume
            quadrature_nodes = c.quadrature_nodes
            quadrature_weights = c.quadrature_weights
            del c
        if cell_shape == "TRIANGLE":
            c = Triangle(vertices, polynomial_order)
            centroid = c.centroid
            volume = c.volume
            quadrature_nodes = c.quadrature_nodes
            quadrature_weights = c.quadrature_weights
            del c
        if cell_shape == "POLYGON":
            c = Polygon(vertices, polynomial_order)
            centroid = c.centroid
            volume = c.volume
            quadrature_nodes = c.quadrature_nodes
            quadrature_weights = c.quadrature_weights
            del c
        if cell_shape == "TETRAHEDRON":
            c = Tetrahedron(vertices, polynomial_order)
            centroid = c.centroid
            volume = c.volume
            quadrature_nodes = c.quadrature_nodes
            quadrature_weights = c.quadrature_weights
            del c
        if cell_shape == "POLYHEDRON":
            c = Polyhedron(vertices, connectivity_matrix, polynomial_order)
            centroid = c.centroid
            volume = c.volume
            quadrature_nodes = c.quadrature_nodes
            quadrature_weights = c.quadrature_weights
            del c
        super().__init__(centroid, volume, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_cell_shape(vertices: Mat) -> str:
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
        problem_dimension = vertices.shape[1]
        if number_of_vertices == 2 and problem_dimension == 1:
            cell_shape = "SEGMENT"
        elif number_of_vertices == 3 and problem_dimension == 2:
            cell_shape = "TRIANGLE"
        elif number_of_vertices == 4 and problem_dimension == 2:
            cell_shape = "QUADRANGLE"
        elif number_of_vertices > 4 and problem_dimension == 2:
            cell_shape = "POLYGON"
        elif number_of_vertices == 4 and problem_dimension == 3:
            cell_shape = "TETRAHEDRON"
        elif number_of_vertices == 6 and problem_dimension == 3:
            cell_shape = "PRISM"
        elif number_of_vertices == 8 and problem_dimension == 3:
            cell_shape = "HEXAHEDRON"
        elif number_of_vertices > 8 and problem_dimension == 3:
            cell_shape = "POLYHEDRON"
        else:
            raise NameError("no match")
        return cell_shape
