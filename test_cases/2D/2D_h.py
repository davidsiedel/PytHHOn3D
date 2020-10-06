import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab

from test_cases import context

from parsers.geof_parser import parse_geof_file as parse_mesh
from parsers.element_types import C_cf_ref
from bases.basis import Basis
from bases.monomial import ScaledMonomial
from core.face import Face
from core.cell import Cell
from core.operators.operator import Operator
from core.operators.hdg import HDG

from pythhon3d import build, solve

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
rc("text", usetex=True)

sizes = [5, 10, 15, 20, 25, 30, 35, 40]
# orders = [1, 2]
orders = [0, 1]
# orders = [3]
# sizes = [5, 10, 15, 20]
# sizes = [5, 15, 25, 35]
sizes = [5, 10, 15, 20, 25, 30, 35, 40]
sizes = [5, 10, 15, 25, 30, 35, 40]
# sizes = [5]
e_list_order = []
for order in orders:
    e_0 = 0
    n_0 = 0
    e_list = []
    h_list = []
    for n in sizes:
        print("order : {}, size : {}".format(order, n))
        d = 2
        face_polynomial_order = order
        cell_polynomial_order = order
        field_dimension = 1
        stabilization_parameter = 1.0  # K2
        mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/2D/{}.geof".format(n)
        operator_type = "HDG"
        operator_type = "HHO"
        #
        pressure_left = [None]
        pressure_right = [None]
        pressure_top = [None]
        pressure_bottom = [None]
        #
        displacement_left = [lambda x: 0.0]
        displacement_right = [lambda x: 0.0]
        displacement_top = [None]
        displacement_bottom = [None]
        load = [lambda x: np.sin(2.0 * np.pi * x[0])]
        #
        boundary_conditions = {
            "RIGHT": (displacement_right, pressure_right),
            "LEFT": (displacement_left, pressure_left),
            "TOP": (displacement_top, pressure_top),
            "BOTTOM": (displacement_bottom, pressure_bottom),
        }
        # --------------------------------------------------------------------------------------------------------------
        (
            vertices,
            faces,
            cells,
            operators,
            cells_faces_connectivity_matrix,
            cells_vertices_connectivity_matrix,
            faces_vertices_connectivity_matrix,
            nsets,
            flags,
            cell_basis_l,
            cell_basis_k,
            face_basis_k,
            unknown,
        ) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
        # --------------------------------------------------------------------------------------------------------------
        d = unknown.problem_dimension
        tangent_matrices = [np.eye(2) for i in range(len(cells))]
        (
            (vertices, unknowns_at_vertices),
            (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
            (vertices, f_unknowns_at_vertices),
            (x_cell_list, x_faces_list),
        ) = solve(
            vertices,
            faces,
            cells,
            operators,
            cells_faces_connectivity_matrix,
            cells_vertices_connectivity_matrix,
            faces_vertices_connectivity_matrix,
            nsets,
            flags,
            cell_basis_l,
            cell_basis_k,
            face_basis_k,
            unknown,
            tangent_matrices,
            stabilization_parameter,
            boundary_conditions,
            load,
        )
        e = 0
        anal_sol = lambda x: -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * x[0]))
        error_at_quadrature_points = np.zeros((quadrature_points.shape[0],))
        for i, quadrature_point in enumerate(quadrature_points):
            local_error = (
                quadrature_weights[i] * (unknowns_at_quadrature_points[0][i] - anal_sol(quadrature_point)) ** 2
            )
            e += local_error
            error_at_quadrature_points[i] += local_error
        e = np.sqrt(e)
        h = 1.0 / float(n)
        if n == 5:
            e_0 = e
            h_0 = h
        n_e = e / e_0
        n_h = h / h_0
        e_list.append(n_e)
        h_list.append(n_h)
        # -----
        x, y = quadrature_points.T
        col_levels = 10
        vmaxx = 0.032
        vminn = -0.032
        levels = np.linspace(vminn, vmaxx, col_levels)
        # plt.tricontourf(x, y, unknowns_at_quadrature_points[0], cmap=cm.binary, levels=levels)
        plt.tricontourf(x, y, error_at_quadrature_points, cmap=cm.binary, levels=levels)
        plt.tricontourf(x, y, error_at_quadrature_points, cmap=cm.binary)
        plt.tricontourf(x, y, unknowns_at_quadrature_points[0], cmap=cm.binary)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.colorbar()
        # plt.show()
        plt.close()
    e_list_order.append(e_list)
# plt.plot([np.log(h_i) for h_i in h_list], [np.log(e_i) for e_i in e_list])
# plt.scatter([np.log(h_i) for h_i in h_list], [np.log(e_i) for e_i in e_list])
# plt.show()

fig, ax = plt.subplots()

for i, order in enumerate(orders):

    if order == 1:
        color = "black"
    else:
        color = "grey"
    ax.plot(
        [np.log(h_i) for h_i in h_list],
        [np.log(e_i) for e_i in e_list_order[i]],
        label="logartithmic $L^2$-error for $k = {}$".format(order),
        color=color,
    )
    ax.plot(
        [np.log(h_i) for h_i in h_list],
        # [np.log(h_i ** (order + 1)) for h_i in h_list],
        # label="${}\log(h/h_0)$".format(order + 1),
        [np.log(h_i ** (order + 1)) for h_i in h_list],
        label="${}\log(h/h_0)$".format(order + 1),
        color=color,
        dashes=[2, 2],
    )
    # ax.scatter(
    #     [np.log(float(nums[0]) / float(num)) for num in nums], [np.log(xi) for xi in RMS_1_n], color="black",
    # )
    ax.scatter(
        [np.log(h_i) for h_i in h_list], [np.log(e_i) for e_i in e_list_order[i]], color=color,
    )
ax.set_xlabel("$\log(h/h_0)$ with $h_0 = 1/5$")
ax.set_ylabel("logartithmic $L^2$-error")
plt.legend()
plt.grid(True)
plt.show()
