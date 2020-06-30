from drafts import context
import numpy as np

from scipy import integrate
from quadratures.dunavant import DunavantRule

nodes = np.array([[0.0, 1.2], [2.8, 0.0], [0.0, 0.0]])
nodes = np.array([[1.0, 1.0], [1.2, 2.2], [1.8, 1.0]])
nodes = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
volume = (2.8 * 1.2) / 2.0
volume = 1.0
polynomial_orders = range(1, 6)

quad_ps, quad_ws = [], []
for polynomial_order in polynomial_orders:
    quad_p, quad_w = DunavantRule.get_triangle_quadrature(nodes, volume, polynomial_order)
    quad_ps.append(quad_p)
    quad_ws.append(quad_w)

functions = [
    # (lambda y, x: 2.0 * x + 3.0 * y + 7.0),
    (lambda y, x: 1.0 * x + 1.0 * y + 0.0),
    (lambda y, x: x ** 2 + 6.0 * x + 3.0 * y ** 2 - 2.0 * y + 2.0 * x * y - 2.0),
    # (lambda y, x: x ** 2 + 6.0 * x + 3.0 * y ** 2 - 2.0 * y - 2.0),
    (lambda y, x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x - 1.0 + 4.0 * y ** 3),
    (lambda y, x: 7.0 * x ** 4 - 4.0 * x ** 3 + 5.0 * x ** 2 - 1.0 * x + 9.0 + 5.0 * y ** 4),
    (lambda y, x: 2.0 * x ** 5 - 1.0 * x ** 4 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x - 6.0 + 6.0 * y ** 5),
]

integs = [
    np.sum([w * functions[k - 1](x[1], x[0]) for x, w in zip(quad_p, quad_w)])
    for k, quad_p, quad_w in zip(range(1, 6), quad_ps, quad_ws)
]
integrals = [
    integrate.dblquad(functions[k - 1], 0.0, 1.0, lambda x: 0.0, lambda x: x)[0]
    + integrate.dblquad(functions[k - 1], 1.0, 2.0, lambda x: 0.0, lambda x: 2.0 - x)[0]
    for k in range(1, 6)
]

print(integs)
print(integrals)

polynomial_orders = range(1, 6)

functions = [
    (lambda x: 2.0 * x + 3.0),
    (lambda x: x ** 2 + 6.0 * x - 2.0),
    (lambda x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x - 1.0),
    (lambda x: 7.0 * x ** 4 - 4.0 * x ** 3 + 5.0 * x ** 2 - 1.0 * x + 9.0),
    (lambda x: 2.0 * x ** 5 - 1.0 * x ** 4 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x - 6.0),
]

test_data = []
num_data = []
for k in polynomial_orders:
    test_data.append((k, integrate.quad(functions[k - 1], 1.1, 2.2)[0]))
    vertices = np.array([[1.1], [2.2]])
    volume = 1.1
    quadrature_points, quadrature_weights = DunavantRule.get_segment_quadrature(vertices, volume, k)
    numerical_integral = np.sum(
        [
            quadrature_weight * functions[k - 1](quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(quadrature_points, quadrature_weights)
        ]
    )
    num_data.append(numerical_integral)
print(test_data)
print(num_data)
