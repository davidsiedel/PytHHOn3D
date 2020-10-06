import argparse
import numpy as np
import matplotlib.pyplot as plt

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

num = 20
d = 1
face_polynomial_order = 1
cell_polynomial_order = 1
field_dimension = 1
stabilization_parameter = 1.0
stabilization_parameter = 1.0  # K2
mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c1d2_{}.geof".format(num)
operator_type = "HHO"
#
# pressure_left = [lambda x: 0.0]
# pressure_right = [lambda x: 0.0]
pressure_left = [None]
pressure_right = [None]
# pressure_right = [lambda x: 1.0]
#
displacement_left = [lambda x: 0.0]
displacement_right = [lambda x: 0.0]
# displacement_right = [None]
#
load = [lambda x: np.sin(2.0 * np.pi * x[0])]
# load = [lambda x: 0.0]
#
boundary_conditions = {
    "RIGHT": (displacement_right, pressure_right),
    "LEFT": (displacement_left, pressure_left),
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
    nsets_faces,
    cell_basis_l,
    cell_basis_k,
    face_basis_k,
    unknown,
) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
# ------------------------------------------------------------------------------------------------------------------
d = unknown.problem_dimension
tangent_matrices = [np.eye(d ** 2) for i in range(len(cells))]
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
    nsets_faces,
    cell_basis_l,
    cell_basis_k,
    face_basis_k,
    unknown,
    tangent_matrices,
    stabilization_parameter,
    boundary_conditions,
    load,
)
# ------------------------------------------------------------------------------------------------------------------
analytical_segmentation = np.linspace(0.0, 1.0, 1000, endpoint=True)
geometrical_segmentation = np.linspace(0.0, 1.0, num, endpoint=True)
# ------------------------------------------------------------------------------------------------------------------
cell_sol = lambda x, x_T, v_T, u: np.sum([u[i] * ((x - x_T) / v_T) ** i for i in range(cell_polynomial_order + 1)])
# ------------------------------------------------------------------------------------------------------------------
# plt.plot(x_axis_v, [-(1.0 / (2.0 * np.pi) ** 2) * load[0](x) for x in x_axis_v], label="analytical")
anal_sol = lambda x: -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * x))
# anal_sol = lambda x: x
fig, ax = plt.subplots()
x_anal = analytical_segmentation
y_anal = [anal_sol(xi) for xi in x_anal]
ax.plot(x_anal, y_anal, color="black", linewidth=1)
for el_index in range(len(geometrical_segmentation) - 1):
    b_left = geometrical_segmentation[el_index]
    b_right = geometrical_segmentation[el_index + 1]
    # ---------------
    n_left = int(1000.0 * b_left)
    n_right = int(1000.0 * b_right)
    # ---------------
    if n_left > 1000:
        n_left = 1000
    if n_left < 0:
        n_left = 0
    # ---------------
    if n_right > 1000:
        n_right = 1000
    if n_right < 0:
        n_right = 0
    # ---------------
    x = analytical_segmentation[n_left:n_right]
    y = []
    for xi in x:
        v_T = b_right - b_left
        x_T = el_index * v_T + 0.5 * v_T
        yi = cell_sol(xi, x_T, v_T, x_cell_list[el_index])
        y.append(yi)
    # y = [cell_sol(xi, x_cell_list[el_index]) for xi in x]
    ax.plot(x, y, color="b")
    # ---------------
    f_unknowns_at_vertices
    ax.scatter(b_left, x_faces_list[el_index][0], color="g")
    ax.scatter(b_right, x_faces_list[el_index][1], color="g")
ax.set_xlabel("Position on the mesh")
ax.set_ylabel("Solution")
plt.show()
