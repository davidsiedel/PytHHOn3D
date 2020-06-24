import numpy as np
from core.face import Face
from core.cell import Cell
from core.element import Element
from core.basis import Basis
from core.boundary import Boundary

from scipy.linalg import solve

k_Q = 4
d = 1
k = 1
num_points = 20
# MESH
mesh = np.linspace(0.0, 1.0, num_points, endpoint=True)
# BOUNDARY
u_d = lambda x: [0]
t_d = lambda x: [0]
b_left = Boundary("LEFT", [0], u_d, t_d)
b_right = Boundary("RIGHT", [num_points - 1], u_d, t_d)
# INTERNAL LOAD
internal_load = lambda x: [
    np.cos(np.pi * x[0]),
]
# internal_load = lambda x: [10.0]
# FACES
faces = (
    [Face(np.array([[mesh[0]]]), k_Q, b_left)]
    + [Face(np.array([[node]]), k_Q) for node in mesh[1 : num_points - 1]]
    + [Face(np.array([[mesh[-1]]]), k_Q, b_right)]
)
# CELLS
cells_nodes = [np.array([[mesh[i - 1]], [mesh[i]]]) for i in range(1, len(mesh))]
C_cf = np.array([[i - 1, i] for i in range(1, len(mesh))])
cells = [
    Cell(
        cells_nodes[i],
        [faces[i], faces[i + 1]],
        np.array([[cells_nodes[i][0]], [cells_nodes[i][1]]]),
        internal_load,
        k_Q,
    )
    for i in range(len(cells_nodes))
]
# BASIS
cell_basis = Basis(k, d)
face_basis = Basis(k, d - 1)
# ELEMENTS
elements = [
    Element(cells[i], [faces[i], faces[i + 1]], cell_basis, face_basis, d)
    for i in range(len(cells))
]
# DIRICHLET WITH LAGRANGE MULTIPLIERS
B = []
total_system_size = face_basis.dim * len(faces)
print(len(faces))
for boundary in [b_left, b_right]:
    for face_index in boundary.faces_index:
        face = faces[face_index]
        b_mat = np.zeros((face_basis.dim, total_system_size))
        b_mat[
            :, face_index * face_basis.dim : (face_index + 1) * face_basis.dim
        ] += np.eye(face_basis.dim)
        B.append(b_mat)
B = np.concatenate(B, axis=0)
# ASSEMBLY
A_tot = np.zeros((total_system_size, total_system_size))
b_tot = np.zeros((total_system_size,))
for elem in elements:
    print("b_cond : {}".format(elem.b))
    for C_cf_loc in C_cf:
        for u, m in enumerate(C_cf_loc):
            b_tot[m * face_basis.dim : (m + 1) * face_basis.dim] += elem.b_cond[
                u * face_basis.dim : (u + 1) * face_basis.dim
            ]
            for v, n in enumerate(C_cf_loc):
                # print("-----------")
                # print(elem.a_cond.shape)
                # print(elem.b_cond.shape)
                # print("-----------")
                # print("u : {}".format(u))
                # print("v : {}".format(v))
                # print("m : {}".format(m))
                # print("n : {}".format(n))
                # print("-----------")
                A_tot[
                    m * face_basis.dim : (m + 1) * face_basis.dim,
                    n * face_basis.dim : (n + 1) * face_basis.dim,
                ] += elem.a_cond[
                    u * face_basis.dim : (u + 1) * face_basis.dim,
                    v * face_basis.dim : (v + 1) * face_basis.dim,
                ]
# ASSEMBLY WITH LAGRANGE
A_tot_augmented = np.concatenate([A_tot, B.T, B.T], axis=1)
I_mat_tot = np.zeros((2 * B.shape[0], 2 * B.shape[0]))
I_mat = np.eye(B.shape[0])
for i in range(len([b_left, b_right])):
    I_mat_tot[: B.shape[0], : B.shape[0]] += I_mat
    I_mat_tot[: B.shape[0], B.shape[0] :] += -I_mat
    I_mat_tot[B.shape[0] :, : B.shape[0]] += -I_mat
    I_mat_tot[B.shape[0] :, B.shape[0] :] += I_mat
temp = np.concatenate((B, B), axis=0)
temp = np.concatenate((temp, I_mat_tot), axis=1)
A_tot_augmented = np.concatenate((A_tot_augmented, temp), axis=0)
h = np.full((2 * B.shape[0],), 0.0)
b_tot_augmented = np.concatenate((b_tot, h))
# print(total_system_size)
# print(b_tot_augmented.shape)
# print(A_tot_augmented.shape)
# solve(A_tot_augmented, b_tot_augmented)
u_d = solve(A_tot_augmented, b_tot_augmented)[:total_system_size]

import matplotlib.pyplot as plt

plt.plot(mesh, u_d)
plt.show()
print("-----UD-----")
print(u_d)
