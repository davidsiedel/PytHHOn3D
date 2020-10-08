import os
from typing import List
from typing import Callable
import numpy as np
from numpy import ndarray as Mat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import rc

from tests import context
from tests.context import source

from pythhon3d import build, solve

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
rc("text", usetex=True)


def solve_2D_poisson_problem(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    field_dimension = 1
    mesh_file = os.path.join(source, "meshes/2D/c2d3_{}.geof".format(number_of_elements))
    # ------------------------------------------------------------------------------------------------------------------
    pressure_left = [None]
    pressure_right = [None]
    pressure_top = [None]
    pressure_bottom = [None]
    # ------------------------------------------------------------------------------------------------------------------
    displacement_left = [lambda x: 0.0]
    displacement_right = [lambda x: 0.0]
    displacement_top = [None]
    displacement_bottom = [None]
    load = [lambda x: np.sin(2.0 * np.pi * x[0])]
    # ------------------------------------------------------------------------------------------------------------------
    boundary_conditions = {
        "RIGHT": (displacement_right, pressure_right),
        "LEFT": (displacement_left, pressure_left),
        "TOP": (displacement_top, pressure_top),
        "BOTTOM": (displacement_bottom, pressure_bottom),
    }
    # ------------------------------------------------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------------------------------------------------
    d = unknown.problem_dimension
    tangent_matrices = [np.eye(2) for i in range(len(cells))]
    # ------------------------------------------------------------------------------------------------------------------
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
    return (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    )


def compute_2D_error(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve_2D_poisson_problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    anal_sol = lambda x: -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * x[0]))
    error = 0.0
    for i in range(len(quadrature_points)):
        error += (anal_sol(quadrature_points[i]) - unknowns_at_quadrature_points[0][i]) * quadrature_weights[i]
    return error, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter


def plot_2D_error(
    h_list: List,
    e_list: List,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    expected_convergence_rate: int,
    ax,
    color,
):
    h_0 = h_list[0]
    e_0 = e_list[0]
    ax.plot(
        [np.log(h_i / h_0) for h_i in h_list],
        [np.log(e_i / e_0) for e_i in e_list],
        label="logartithmic $L^2$-error for $k = {}, l = {}$".format(face_polynomial_order, cell_polynomial_order),
        color=color,
    )
    ax.plot(
        [np.log((h_i / h_0)) for h_i in h_list],
        [np.log((h_i / h_0) ** (face_polynomial_order + expected_convergence_rate)) for h_i in h_list],
        label="${}\log(h/h_0)$".format(face_polynomial_order + expected_convergence_rate),
        color=color,
        dashes=[2, 2],
    )
    ax.scatter(
        [np.log(h_i / h_0) for h_i in h_list], [np.log(e_i / e_0) for e_i in e_list], color=color,
    )
    return


def plot_2D_error_map(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
    fig,
    ax,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve_2D_poisson_problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    vmax = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.75))
    x, y = quadrature_points.T
    error = unknowns_at_quadrature_points[0] - np.array(
        [-(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * quad_p)) for quad_p in x]
    )
    error = np.abs(error) / vmax * 100.0
    data = ax.tricontourf(x, y, error, cmap=cm.binary)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title("Error")
    cbar = fig.colorbar(data, ax=ax)
    cbar.set_label("Relative error (percent)", rotation=270, labelpad=15.0)
    return


def plot_2D_solution_map(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
    fig,
    ax,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve_2D_poisson_problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    vmax = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.75))
    x, y = quadrature_points.T
    data = ax.tricontourf(x, y, unknowns_at_quadrature_points[0], cmap=cm.binary)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title(
        "{}, $k = {}, l = {}$, stab $= {}$".format(
            operator_type, face_polynomial_order, cell_polynomial_order, stabilization_parameter
        )
    )
    cbar = fig.colorbar(data, ax=ax)
    cbar.set_label("{} solution".format(operator_type), rotation=270, labelpad=15.0)
    return


def plot_2D_analytical_map(fig, ax):
    x = np.linspace(0.0, 1.0, 1000)
    y = np.linspace(0.0, 1.0, 1000)
    X, Y = np.meshgrid(x, y)
    Z = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * X))
    vmax = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.75))
    vmin = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.25))
    col_levels = 10
    CS = ax.contourf(X, Y, Z, col_levels, vmin=vmin, vmax=vmax, cmap=cm.binary)
    m = plt.cm.ScalarMappable(cmap=cm.binary)
    m.set_array(Z)
    m.set_clim(vmin, vmax)
    cbar = fig.colorbar(m, boundaries=np.linspace(vmin, vmax, col_levels), ax=ax)
    cbar.set_label("Analytical solution", rotation=270, labelpad=15.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title("Analytical Solution")
    return

