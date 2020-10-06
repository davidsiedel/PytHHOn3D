import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path

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

rms_list = []
nums = range(20, 105, 5)
for num in nums:
    # ------------------------------------------------------------------------------------------------------------------
    # Defining the number of elements on the unit segment
    # ------------------------------------------------------------------------------------------------------------------
    # num = 7
    # ------------------------------------------------------------------------------------------------------------------
    # Generating the mesh file
    # ------------------------------------------------------------------------------------------------------------------
    current_folder = Path(os.path.dirname(os.path.abspath(__file__)))
    source = current_folder.parent.parent
    meshes_folder = os.path.join(source, "meshes")
    file_name = "c1d2_{}.geof".format(num)
    file_path = os.path.join(meshes_folder, file_name)
    # ------------------------------------------------------------------------------------------------------------------
    with open(file_path, "w") as mesh_file:
        mesh_file.write("***geometry\n**node\n{} 1\n".format(num))
        segmentation = np.linspace(0.0, 1.0, num)
        for i, point in enumerate(segmentation):
            mesh_file.write("{} {}\n".format(i + 1, point))
        mesh_file.write("**element\n{}\n".format(num - 1))
        for i, point in enumerate(segmentation[:-1]):
            mesh_file.write("{} c1d2 {} {}\n".format(i + 1, i + 1, i + 2))
        mesh_file.write("***group\n**nset LEFT\n1\n**nset RIGHT\n{}\n***return".format(num))
    # ------------------------------------------------------------------------------------------------------------------
    # Defining the problem
    # ------------------------------------------------------------------------------------------------------------------
    d = 1
    face_polynomial_order = 1
    cell_polynomial_order = 1
    field_dimension = 1
    stabilization_parameter = 1.0
    mesh_file = "/Users/davidsiedel/Projects/PytHHOn3D/meshes/c1d2_{}.geof".format(num)
    operator_type = "HDG"
    pressure_left = [None]
    pressure_right = [None]
    #
    displacement_left = [lambda x: 0.0]
    displacement_right = [lambda x: 0.0]
    #
    load = [lambda x: np.sin(2.0 * np.pi * x[0])]
    #
    boundary_conditions = {
        "RIGHT": (displacement_right, pressure_right),
        "LEFT": (displacement_left, pressure_left),
    }
    # ------------------------------------------------------------------------------------------------------------------
    # Building the HHO version of the problem
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
    # Defining the tnagent operator
    # ------------------------------------------------------------------------------------------------------------------
    d = unknown.problem_dimension
    tangent_matrices = [np.eye(d ** 2) for i in range(len(cells))]
    # ------------------------------------------------------------------------------------------------------------------
    # Solving the problem
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
    # Post-treatment
    # ------------------------------------------------------------------------------------------------------------------
    analytical_segmentation = np.linspace(0.0, 1.0, 1000, endpoint=True)
    geometrical_segmentation = np.linspace(0.0, 1.0, num, endpoint=True)
    # ------------------------------------------------------------------------------------------------------------------
    cell_sol = lambda x, x_T, v_T, u: np.sum([u[i] * ((x - x_T) / v_T) ** i for i in range(cell_polynomial_order + 1)])
    # ------------------------------------------------------------------------------------------------------------------
    anal_sol = lambda x: -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * x))
    fig, ax = plt.subplots()
    x_anal = analytical_segmentation
    y_anal = [anal_sol(xi) for xi in x_anal]
    ax.plot(x_anal, y_anal, color="black", linewidth=1)
    x_hho, y_hho = [], []
    # ------------------------------------------------------------------------------------------------------------------
    for el_index in range(len(geometrical_segmentation) - 1):
        b_left = geometrical_segmentation[el_index]
        b_right = geometrical_segmentation[el_index + 1]
        # --------------------------------------------------------------------------------------------------------------
        n_left = int(1000.0 * b_left)
        n_right = int(1000.0 * b_right)
        # --------------------------------------------------------------------------------------------------------------
        if n_left > 1000:
            n_left = 1000
        if n_left < 0:
            n_left = 0
        # --------------------------------------------------------------------------------------------------------------
        if n_right > 1000:
            n_right = 1000
        if n_right < 0:
            n_right = 0
        # --------------------------------------------------------------------------------------------------------------
        x = analytical_segmentation[n_left:n_right]
        y = []
        for xi in x:
            v_T = b_right - b_left
            x_T = el_index * v_T + 0.5 * v_T
            yi = cell_sol(xi, x_T, v_T, x_cell_list[el_index])
            y.append(yi)
        ax.plot(x, y, color="b")
        # --------------------------------------------------------------------------------------------------------------
        ax.scatter(b_left, x_faces_list[el_index][0], color="g")
        ax.scatter(b_right, x_faces_list[el_index][1], color="g")
        # --------------------------------------------------------------------------------------------------------------
        x_hho += list(x)
        y_hho += list(y)
    # ------------------------------------------------------------------------------------------------------------------
    y_anal = np.interp(x_hho, x_anal, y_anal)
    rms = np.sqrt(1.0 / (len(x_hho)) * np.sum([(ya - yh) ** 2 for ya, yh in zip(y_anal, y_hho)]))
    # print(rms)
    rms_list.append(rms)
    # ------------------------------------------------------------------------------------------------------------------
    ax.set_xlabel("Position on the mesh")
    ax.set_ylabel("Solution")
    # plt.show()
plt.close()
fig, ax = plt.subplots()
ax.plot(nums, rms_list)
print("RMS : {}".format(rms_list))
print("nums : {}".format([num for num in nums]))
ax.set_xlim(20.0, 100.0)
ax.set_ylim(0.0, 0.0008)
ax.set_xlabel("number of elements")
ax.set_ylabel("RMS error")
plt.show()
