import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable


class Boundary:
    def __init__(
        self,
        name: str,
        faces_index: List[int],
        imposed_dirichlet: Callable,
        imposed_neumann: Callable,
    ):
        """
        ejbchbjs
        """
        self.name = name
        self.faces_index = faces_index
        self.imposed_dirichlet = imposed_dirichlet
        self.imposed_neumann = imposed_neumann
        return
