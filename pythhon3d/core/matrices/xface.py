import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from shapes.domain import Domain
from core.face import Face
from core.unknown import Unknown
from core.integration import Integration
from bases.basis import Basis

# from shapes.segment import Segment
# from shapes.triangle import Triangle
# from shapes.polygon import Polygon
# from shapes.tetrahedron import Tetrahedron
# from shapes.polyhedron import Polyhedron

class Xface:
    def __init__(self, face: Face, cell_basis_l: Basis, cell_basis_k: Basis, cell_basis_k1: Basis, unknown: Unknown)::
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Xface class 
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - 
        """

