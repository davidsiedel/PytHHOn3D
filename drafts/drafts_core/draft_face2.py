from drafts import context
from core.face2 import Face
import numpy as np
import pytest


polynomial_order = 1
face = Face(np.array([[0.1]]), polynomial_order)  # POINT
face = Face(np.array([[0.0, 0.0], [1.0, 0.0]]), polynomial_order)  # SEGMENT
face = Face(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), polynomial_order)  # TRIANGLE
face = Face(
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), polynomial_order,
)  # POLYGON
