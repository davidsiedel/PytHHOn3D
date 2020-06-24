# from pythhon3d.src.core.basis import Basis
from core.basis import Basis
import numpy as np

k = 2
d = 2
b = Basis(k, d)
print(b.get_gradient_operator(k, d, 0))

point = np.array([0.2])
barycenter = np.array([0.4])
volume = 1.3

f = lambda alpha: (alpha / volume) * ((point[0] - barycenter[0]) / volume) ** (alpha - 1)

print([f(0), f(1), f(2), f(3)])


point = np.array([0.2, 1.5])
barycenter = np.array([0.4, 0.6])
volume = 1.3

g = (
    lambda alpha, beta: (alpha / volume)
    * ((point[0] - barycenter[0]) / volume) ** (alpha - 1)
    * ((point[1] - barycenter[1]) / volume) ** (beta)
)

print(
    [
        g(0, 0),
        g(0, 1),
        g(1, 0),
        g(0, 2),
        g(1, 1),
        g(2, 0),
        g(0, 3),
        g(1, 2),
        g(2, 1),
        g(3, 0),
    ]
)

