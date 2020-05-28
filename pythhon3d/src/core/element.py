import numpy as np
from numpy import ndarray as Mat
from scipy.linalg import block_diag
from scipy.linalg import cholesky, solve_triangular, inv
from typing import List
from HHO.Basis import Basis
from HHO.Face import Face
from HHO.Cell import Cell
from typing import TypeVar

# --------------------------------------------------------------------------------------
# One has the following matrices for the whole mesh:
# N, C_nc, C_nf, C_cf, weights, Nsets, flags
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# CELL
# --------------------------------------------------------------------------------------
# barycenter
# volume
# nodes_Q
# weigh_Q
# signs
# --------------------------------------------------------------------------------------
# FACE
# --------------------------------------------------------------------------------------
# flag
# barycenter
# p_matrix
# normal_vector
# volume
# nodes_Q
# weigh_Q


class Element:
    def __init__(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        d_out: int,
    ):
        self.cell = cell
        self.faces = faces
        self.cell_basis = cell_basis
        self.face_basis = face_basis
        # ------------------------------------------------------------------------------
        # Reading the dimension of the euclidian space.
        # ------------------------------------------------------------------------------
        d = self.cell.barycenter.shape[0]

        self.s_tensor = self.get_stabilization_vector_tensor(
            self.cell, self.faces, self.cell_basis, self.face_basis, d_out
        )
        self.b_tensor = self.get_reconstructed_gradient_tensor(
            self.cell, self.faces, self.cell_basis, self.face_basis, d, d_out
        )

        a_1 = self.get_recontructed_gradient_bilinear_form_tensor(
            self.cell, self.cell_basis, self.b_tensor, d, d_out
        )
        a_2 = self.get_stabilization_bilinear_form_tensor(
            self.faces, self.face_basis, self.s_tensor, d_out
        )
        self.a = a_1 + a_2

        self.b = self.get_second_member_vector(
            self.cell, self.faces, self.cell_basis, self.face_basis, d_out
        )

        self.a_cond, self.b_cond = self.condensate_local_problem(
            self.a, self.b, self.cell_basis
        )

    def get_cell_mass_matrix_in_cell(self, cell: Cell, cell_basis: Basis) -> Mat:
        """
        Returns the mass matrix M in a cell with indices i,j:
        Mij = phi_i(quadrature_node) * phi_j(quadrature_node)
        """
        v = cell.volume
        x_b = cell.barycenter
        cell_mass_matrix_in_cell = np.zeros((cell_basis.dim, cell_basis.dim))
        for x_Q, w_Q in zip(cell.nodes_Q, cell.weigh_Q):
            phi_vector = np.array([cell_basis.get_phi_vector(x_Q, x_b, v)])
            cell_mass_matrix_in_cell_loc = w_Q * phi_vector.T @ phi_vector
            cell_mass_matrix_in_cell += cell_mass_matrix_in_cell_loc
        return cell_mass_matrix_in_cell

    def get_face_mass_matrix_in_face(self, face: Face, face_basis: Basis) -> Mat:
        """
        Returns the mass matrix M in a face with indices i,j:
        Mij = psi_i(quadrature_node) * psi_j(quadrature_node)
        """
        # ------------------------------------------------------------------------------
        # Reading the dimension of the euclidian space.
        # ------------------------------------------------------------------------------
        d = face.barycenter.shape[0]
        # ------------------------------------------------------------------------------
        # Getting the rows to consider to get the projection of d-dimensional nodes onto
        # a planar face.
        # ------------------------------------------------------------------------------
        if d == 3:
            rows = [0, 1]
        elif d == 2 or d == 1:
            rows = [0]
        # ------------------------------------------------------------------------------
        # Initializing the cell mass matrix evaluated on the face hyperplane.
        # ------------------------------------------------------------------------------
        face_mass_matrix_in_face = np.zeros((face_basis.dim, face_basis.dim))
        # ------------------------------------------------------------------------------
        # Changing the reference frame to that of the planar hyperplane supporting the
        # face.
        # ------------------------------------------------------------------------------
        face_nodes_Q_loc = (face.p_matrix.T @ (face.nodes_Q).T).T
        face_barycenter_loc = (face.p_matrix.T @ face.barycenter.T).T
        # ------------------------------------------------------------------------------
        # Reading the dimension of the euclidian space.
        # ------------------------------------------------------------------------------
        for face_node_Q_loc, face_weigh_Q in zip(face_nodes_Q_loc, face.weigh_Q):
            psi_vector = np.array(
                [
                    face_basis.get_phi_vector(
                        face_node_Q_loc[rows], face_barycenter_loc[rows], face.volume,
                    )
                ]
            )
            face_mass_matrix_in_face_loc = face_weigh_Q * psi_vector.T @ psi_vector
            face_mass_matrix_in_face += face_mass_matrix_in_face_loc
        return face_mass_matrix_in_face

    def get_cell_advection_matrix_in_cell(
        self, cell: Cell, cell_basis: Basis, dx: int
    ) -> Mat:
        """
        Returns the advection matrix in a domain with indices i,j:
        Mij = d_phi_i_d_x_j(quadrature_node) * phi_j(quadrature_node)
        """
        v = cell.volume
        x_b = cell.barycenter
        cell_advection_matrix = np.zeros((cell_basis.dim, cell_basis.dim))
        for x_Q, w_Q in zip(cell.nodes_Q, cell.weigh_Q):
            phi_vector = np.array([cell_basis.get_phi_vector(x_Q, x_b, v)])
            d_phi_vector = np.array([cell_basis.get_d_phi_vector(x_Q, x_b, v, dx)])
            cell_mass_advection_loc = w_Q * phi_vector.T @ d_phi_vector
            cell_advection_matrix += cell_mass_advection_loc
        return cell_advection_matrix

    def get_hybrid_mass_matrix_in_face(
        self,
        face: Face,
        cell: Cell,
        face_basis: Basis,
        cell_basis: Basis,
        sign: float = 1.0,
        x: int = -1,
    ) -> Mat:
        """
        Return the hybrid mass matrix in a face with indices i,j:
        Mij = psi_i(quadrature_node) * phi_j(quadrature_node)
        """
        # ------------------------------------------------------------------------------
        # Reading the dimension of the euclidian space.
        # ------------------------------------------------------------------------------
        d = cell.barycenter.shape[0]
        # ------------------------------------------------------------------------------
        # Getting the rows to consider to get the projection of d-dimensional nodes onto
        # a planar face.
        # ------------------------------------------------------------------------------
        if d == 3:
            rows = [0, 1]
        elif d == 2 or d == 1:
            rows = [0]
        # ------------------------------------------------------------------------------
        # Initializing the cell mass matrix evaluated on the face hyperplane.
        # ------------------------------------------------------------------------------
        hybrid_mass_matrix_in_face = np.zeros((cell_basis.dim, face_basis.dim))
        # ------------------------------------------------------------------------------
        # Changing the reference frame to that of the planar hyperplane supporting the
        # face.
        # ------------------------------------------------------------------------------
        face_nodes_Q_loc = (face.p_matrix.T @ (face.nodes_Q).T).T
        face_barycenter_loc = (face.p_matrix.T @ face.barycenter.T).T
        cell_barycenter_loc = (face.p_matrix.T @ cell.barycenter.T).T
        # ------------------------------------------------------------------------------
        # Reading the dimension of the euclidian space.
        # ------------------------------------------------------------------------------
        for face_node_Q_loc, face_weigh_Q in zip(face_nodes_Q_loc, face.weigh_Q):
            phi_vector = np.array(
                [
                    cell_basis.get_phi_vector(
                        face_node_Q_loc, cell_barycenter_loc, cell.volume
                    )
                ]
            )
            psi_vector = np.array(
                [
                    face_basis.get_phi_vector(
                        face_node_Q_loc[rows], face_barycenter_loc[rows], face.volume,
                    )
                ]
            )
            if not x == -1:
                hybrid_mass_matrix_in_face_loc = (
                    face_weigh_Q
                    * phi_vector.T
                    @ psi_vector
                    * sign
                    * face.normal_vector[x]
                )
            else:
                hybrid_mass_matrix_in_face_loc = (
                    face_weigh_Q * phi_vector.T @ psi_vector * sign
                )
            hybrid_mass_matrix_in_face += hybrid_mass_matrix_in_face_loc
        return hybrid_mass_matrix_in_face

    def get_cell_mass_matrix_in_face(
        self, face: Face, cell: Cell, cell_basis: Basis, sign: float = 1.0, x: int = -1,
    ) -> Mat:
        """
        Return the hybrid mass matrix of a face.
        """
        # ------------------------------------------------------------------------------
        # Initializing the cell mass matrix evaluated on the face hyperplane
        # ------------------------------------------------------------------------------
        cell_mass_matrix_in_face = np.zeros((cell_basis.dim, cell_basis.dim))
        # ------------------------------------------------------------------------------
        # Computing the cell mass matrix for each face quadarture node
        # ------------------------------------------------------------------------------
        for (face_node_Q, face_weigh_Q) in zip(face.nodes_Q, face.weigh_Q):
            phi_vector = np.array(
                [cell_basis.get_phi_vector(face_node_Q, cell.barycenter, cell.volume)]
            )
            if not x == -1:
                cell_mass_matrix_in_face_loc = (
                    -face_weigh_Q
                    * phi_vector.T
                    @ phi_vector
                    * sign
                    * face.normal_vector[x]
                )
            else:
                cell_mass_matrix_in_face_loc = (
                    -face_weigh_Q * phi_vector.T @ phi_vector * sign
                )
            cell_mass_matrix_in_face += cell_mass_matrix_in_face_loc
        return cell_mass_matrix_in_face

    def get_reconstructed_gradient_tensor_component(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        dx: int,
    ) -> Mat:
        """
        Return the hybrid mass matrix of a face.
        """
        # ------------------------------------------------------------------------------
        # Computing left side mass matrix
        # ------------------------------------------------------------------------------
        left_matrix_loc = self.get_cell_mass_matrix_in_cell(cell, cell_basis)
        # ------------------------------------------------------------------------------
        # Computing left side mass matrix
        # ------------------------------------------------------------------------------
        right_matrix_loc = np.zeros(
            (cell_basis.dim, cell_basis.dim + len(faces) * face_basis.dim)
        )
        # ------------------------------------------------------------------------------
        # Computing right side advection matrix
        # ------------------------------------------------------------------------------
        cell_advection_matrix_in_cell = self.get_cell_advection_matrix_in_cell(
            cell, cell_basis, dx
        )
        right_matrix_loc[:, : cell_basis.dim] += cell_advection_matrix_in_cell
        # ------------------------------------------------------------------------------
        # Computing right side advection matrix
        # ------------------------------------------------------------------------------
        for i, face in enumerate(faces):
            cell_mass_matrix_in_face = self.get_cell_mass_matrix_in_face(
                face, cell, cell_basis, cell.signs[i], dx
            )
            right_matrix_loc[:, : cell_basis.dim] += cell_mass_matrix_in_face
            hybrid_mass_matrix_in_face = self.get_hybrid_mass_matrix_in_face(
                face, cell, face_basis, cell_basis, cell.signs[i], dx
            )
            right_matrix_loc[
                :,
                cell_basis.dim
                + i * face_basis.dim : cell_basis.dim
                + (i + 1) * face_basis.dim,
            ] += hybrid_mass_matrix_in_face
        reconstructed_gradient_matrix_loc = inv(left_matrix_loc) @ right_matrix_loc
        return reconstructed_gradient_matrix_loc

    def get_reconstructed_gradient_tensor(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        d: int,
        d_out: int,
    ) -> Mat:
        """
        Return the hybrid mass matrix of a face.
        """
        local_system_size = (
            cell_basis.dim,
            cell_basis.dim + len(faces) * face_basis.dim,
        )
        global_matrix = np.zeros(
            (d * d_out * local_system_size[0], d_out * local_system_size[1],)
        )
        count = 0
        if d == 1:
            multi_index_tab = [0]
        elif d == 2:
            multi_index_tab = [0, 2, 3, 1]
        elif d == 3:
            multi_index_tab = [0, 3, 8, 6, 1, 4, 5, 7, 2]
        for x in range(d_out):
            for dx in range(d):
                reconstructed_gradient_matrix_loc = self.get_reconstructed_gradient_tensor_component(
                    cell, faces, cell_basis, face_basis, dx
                )
                global_matrix[
                    multi_index_tab[count]
                    * local_system_size[0] : (multi_index_tab[count] + 1)
                    * local_system_size[0],
                    x * local_system_size[1] : (x + 1) * local_system_size[1],
                ] += reconstructed_gradient_matrix_loc
                count += 1
        return global_matrix

    def get_cell_polynomial_projector_onto_face_matrix(
        self, cell: Cell, face: Face, cell_basis: Basis, face_basis: Basis
    ) -> Mat:
        """
        khgjkhbejbjk
        """
        left_matrix_loc = self.get_face_mass_matrix_in_face(face, face_basis)
        b = self.get_hybrid_mass_matrix_in_face(face, cell, face_basis, cell_basis)
        right_matrix_loc = b.T
        projector_onto_face = inv(left_matrix_loc) @ right_matrix_loc
        return projector_onto_face

    def get_stabilization_vector_tensor_component(
        self, cell: Cell, faces: List[Face], cell_basis: Basis, face_basis: Basis,
    ) -> (Mat, Mat):
        """
        jncjbcj
        """
        total_system_size = (
            face_basis.dim,
            cell_basis.dim + len(faces) * face_basis.dim,
        )
        stabilization_vector_matrix = np.zeros(total_system_size)
        for i, face in enumerate(faces):
            pi = self.get_cell_polynomial_projector_onto_face_matrix(
                cell, face, cell_basis, face_basis
            )
            # ----------------------------------------------------------------------
            # khbkjbkjk
            # pour savoir pourquoi k cell peut etre k face-1 kface ou kface +1,
            # voir cocckburn brifging
            # ----------------------------------------------------------------------
            stabilization_vector_matrix[
                :,
                cell_basis.dim
                + i * face_basis.dim : cell_basis.dim
                + (i + 1) * face_basis.dim,
            ] += -np.eye(face_basis.dim)
            stabilization_vector_matrix[:, : cell_basis.dim] += pi
        return stabilization_vector_matrix

    def get_stabilization_vector_tensor(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        d_out: int,
    ) -> (Mat, Mat):
        """
        jkzebncjkbjb
        """
        local_system_size = (
            face_basis.dim,
            cell_basis.dim + len(faces) * face_basis.dim,
        )
        global_matrix = np.zeros(
            (d_out * local_system_size[0], d_out * local_system_size[1])
        )
        stabilization_vector_matrix = self.get_stabilization_vector_tensor_component(
            cell, faces, cell_basis, face_basis
        )
        x_tab = range(d_out)
        for x in x_tab:
            global_matrix[
                x * local_system_size[0] : (x + 1) * local_system_size[0],
                x * local_system_size[1] : (x + 1) * local_system_size[1],
            ] += stabilization_vector_matrix
        return global_matrix

    def get_recontructed_gradient_bilinear_form_tensor(
        self, cell: Cell, cell_basis: Basis, b_tensor: Mat, d: int, d_out: int,
    ) -> Mat:
        """
        ibzcbsckj
        """
        total_system_size = d * d_out
        local_system_size = (cell_basis.dim, cell_basis.dim)
        global_mass_matrix = np.zeros(
            (
                total_system_size * local_system_size[0],
                total_system_size * local_system_size[1],
            )
        )
        # ------------------------------------------------------------------------------
        # Defining the stifness matrix
        # ------------------------------------------------------------------------------
        lam = 0.0
        mu = 1.0 / 2.0
        lam_mat = np.full((d_out, d_out), lam)
        stifness_mat = 2.0 * mu * np.eye(d_out * d)
        stifness_mat[:d_out, :d_out] += lam_mat
        # ------------------------------------------------------------------------------
        # Defining the elasticity constants
        # ------------------------------------------------------------------------------
        cell_mass_matrix_in_cell = self.get_cell_mass_matrix_in_cell(cell, cell_basis)
        for i in range(total_system_size):
            for j in range(total_system_size):
                global_mass_matrix[
                    i * local_system_size[0] : (i + 1) * local_system_size[0],
                    j * local_system_size[1] : (j + 1) * local_system_size[1],
                ] += (cell_mass_matrix_in_cell * stifness_mat[i, j])
        # for i in range(total_system_size):
        #     global_mass_matrix[
        #         i * local_system_size[0] : (i + 1) * local_system_size[0], :
        #     ] += cell_mass_matrix_in_cell
        # ------------------------------------------------------------------------------
        # Defining the elasticity constants
        # ------------------------------------------------------------------------------
        recontructed_gradient_bilinear_form_tensor = b_tensor.T @ (
            global_mass_matrix @ b_tensor
        )
        # recontructed_gradient_bilinear_form_tensor = (
        #     2 * mu_coef * b_tensor.T @ (global_mass_matrix @ b_tensor)
        # )
        # recontructed_divergence_bilinear_form_tensor = (
        #     lambda_coef * d_tensor.T @ (cell_mass_matrix_in_cell @ d_tensor)
        # )
        return recontructed_gradient_bilinear_form_tensor

    def get_stabilization_bilinear_form_tensor(
        self, faces: List[Face], face_basis: Basis, s_tensor: Mat, d_out: int,
    ) -> Mat:
        """
        ibzcbsckj
        """
        total_system_size = d_out
        local_system_size = (face_basis.dim, face_basis.dim)
        boundary_mass_matrix_in_boundary = np.zeros(
            (
                total_system_size * local_system_size[0],
                total_system_size * local_system_size[1],
            )
        )
        # boundary_mass_matrix_in_boundary = np.zeros((face_basis.dim, face_basis.dim))
        for face in faces:
            face_mass_matrix_in_face = (
                1.0 / face.diameter
            ) * self.get_face_mass_matrix_in_face(face, face_basis)
            for i in range(total_system_size):
                for j in range(total_system_size):
                    first_row = i * local_system_size[0]
                    last_row = (i + 1) * local_system_size[0]
                    first_col = j * local_system_size[1]
                    last_col = (j + 1) * local_system_size[1]
                    boundary_mass_matrix_in_boundary[
                        first_row:last_row, first_col:last_col
                    ] += face_mass_matrix_in_face
        stabilization_bilinear_form_tensor = s_tensor.T @ (
            boundary_mass_matrix_in_boundary @ s_tensor
        )
        return stabilization_bilinear_form_tensor

    def get_cell_internal_load_vector(
        self, cell: Cell, cell_basis: Basis, x: int
    ) -> Mat:
        """
        jcbhezhbkbh
        """
        v = cell.volume
        x_b = cell.barycenter
        # internal_load_vector = np.zeros((cell_basis.dim, 1))
        internal_load_vector = np.zeros((cell_basis.dim,))
        for x_Q, w_Q in zip(cell.nodes_Q, cell.weigh_Q):
            # phi_vector = np.array([cell_basis.get_phi_vector(x_Q, x_b, v)])
            phi_vector = cell_basis.get_phi_vector(x_Q, x_b, v)
            f_vector = np.full((phi_vector.shape), cell.internal_load(x_Q)[x])
            print("f_vector : {}".format(f_vector))
            print("phi_vector : {}".format(phi_vector))
            print("w_Q : {}".format(w_Q))
            internal_load_vector += w_Q * phi_vector * f_vector
        print("internal_load : {}".format(internal_load_vector))
        return internal_load_vector

    def get_face_neumann_vector(self, face: Face, face_basis: Basis, x: int) -> Mat:
        """
        jcbhezhbkbh
        """
        external_load_vector = np.zeros((face_basis.dim,))
        # ------------------------------------------------------------------------------
        # Reading the dimension of the euclidian space.
        # ------------------------------------------------------------------------------
        d = face.barycenter.shape[0]
        # ------------------------------------------------------------------------------
        # Getting the rows to consider to get the projection of d-dimensional nodes onto
        # a planar face.
        # ------------------------------------------------------------------------------
        if d == 3:
            rows = [0, 1]
        elif d == 2 or d == 1:
            rows = [0]
        # ------------------------------------------------------------------------------
        # Changing the reference frame to that of the planar hyperplane supporting the
        # face.
        # ------------------------------------------------------------------------------
        face_nodes_Q_loc = (face.p_matrix.T @ (face.nodes_Q).T).T
        face_barycenter_loc = (face.p_matrix.T @ face.barycenter.T).T
        # ------------------------------------------------------------------------------
        # Reading the dimension of the euclidian space.
        # ------------------------------------------------------------------------------
        for face_node_Q_loc, face_weigh_Q in zip(face_nodes_Q_loc, face.weigh_Q):
            psi_vector = face_basis.get_phi_vector(
                face_node_Q_loc[rows], face_barycenter_loc[rows], face.volume,
            )
            t_vector = np.full(
                (psi_vector.shape), face.boundary.imposed_neumann(face_node_Q_loc)[x]
            )
            external_load_vector += face_weigh_Q * psi_vector * t_vector
        return external_load_vector

    def get_second_member_vector(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis: Basis,
        face_basis: Basis,
        d_out: int,
    ) -> Mat:
        """
        kncfjnzejknekz
        """
        local_system_size = cell_basis.dim + len(faces) * face_basis.dim
        total_system_size = d_out * local_system_size
        # global_vector = np.zeros((total_system_size, 1))
        global_vector = np.zeros((total_system_size,))
        for x in range(d_out):
            global_vector[
                x * local_system_size : x * local_system_size + cell_basis.dim
            ] += self.get_cell_internal_load_vector(cell, cell_basis, x)
            for i, face in enumerate(faces):
                if not face.boundary is None:
                    global_vector[
                        x * local_system_size
                        + i * face_basis.dim : x * local_system_size
                        + (i + 1) * face_basis.dim
                    ] += self.get_face_neumann_vector(face, face_basis, x)
        return global_vector

    def condensate_local_problem(self, a: Mat, b: Mat, cell_basis: Basis) -> Mat:
        """
        hbckjqbcjb
        """
        a_T_T = a[: cell_basis.dim, : cell_basis.dim]
        a_T_dT = a[: cell_basis.dim, cell_basis.dim :]
        a_dT_T = a[cell_basis.dim :, : cell_basis.dim]
        a_dT_dT = a[cell_basis.dim :, cell_basis.dim :]
        b_T = b[: cell_basis.dim]
        b_dT = b[cell_basis.dim :]
        m = a_dT_T @ inv(a_T_T) @ a_T_dT
        a_cond = a_dT_dT - m
        b_cond = b_dT - m @ b_T
        return a_cond, b_cond

    def decondensate_local_prolem(
        self, a: Mat, b: Mat, cell_basis: Basis, face_basis: Basis
    ) -> Mat:
        """
        jebzkcj
        """

        return
