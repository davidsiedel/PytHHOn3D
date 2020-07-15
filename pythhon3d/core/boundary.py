import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable


class Boundary:
    def __init__(
        self, name: str, faces_index: List[int], displacement: List[Callable], pressure: List[Callable],
    ):
        """
        ejbchbjs
        """
        self.name = name
        self.faces_index = faces_index
        self.displacement = displacement
        self.pressure = pressure
