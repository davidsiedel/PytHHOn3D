import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
from tests.test_error.analysis_2d import *

operator_type = "HDG"
stabilization_parameter = 1.0
expected_convergence_rate = 2

fig, ax = plt.subplots()

orders = [(1, 2), (2, 3)]
colors = [(flt, flt, flt) for flt in np.linspace(0.0, 0.5, len(orders), endpoint=True)]

for (face_polynomial_order, cell_polynomial_order), color in zip(orders, colors):
    num_list = range(5, 35, 5)
    h_list = [1 / float(num) for num in num_list]
    e_list = []
    print(h_list)
    for number_of_elements in num_list:
        error, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter = compute_2D_error(
            number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
        )
        e_list.append(error)
    print(e_list)
    plot_2D_error(h_list, e_list, face_polynomial_order, cell_polynomial_order, expected_convergence_rate, ax, color)

ax.set_xlabel("$\log(h/h_0)$ with $h_0 = 1/5$")
ax.set_ylabel("logartithmic $L^2$-error")
plt.legend()
plt.grid(True)
plt.show()
