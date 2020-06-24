import numpy as np
from numpy import ndarray as Mat
from core.quadrature import *
from core.face import Face
from typing import List
from typing import Callable

# ------------------------------------------------------------------------------
# One has the following matrices for the whole mesh:
# N, C_nc, C_nf, C_cf, weights, Nsets, flags
# ------------------------------------------------------------------------------
class Cell:
    def __init__(
        self,
        cell_nodes: Mat,
        faces: List[Face],
        faces_nodes: List[Mat],
        internal_load: Callable,
        k_Q: int,
    ):
        """
        """
        self.barycenter = self.get_cell_barycenter(cell_nodes)
        (
            self.volume,
            self.nodes_Q,
            self.weigh_Q,
            self.signs,
        ) = self.get_cell_integration_data(faces, faces_nodes, k_Q)
        self.internal_load = internal_load
        return

    def get_cell_barycenter(self, nodes: Mat) -> Mat:
        """
        Getting the barycenter of any d dimensional spatial domain given its nodes
        Returns :
        - barycenter : the cell barycenter
        """
        barycenter = [np.mean(nodes[:, i]) for i in range(nodes.shape[1])]
        return np.array(barycenter)

    def get_vector_to_face(self, face: Face, cell_barycenter: Mat) -> Mat:
        """
        """
        vector_to_face = (face.p_matrix @ (cell_barycenter - face.barycenter).T).T
        return vector_to_face

    def get_cell_integration_data(
        self, faces: List[Face], faces_nodes: List[Mat], k_Q: int
    ) -> (float, Mat, Mat, Mat):
        """
        """
        cell_volume = 0.0
        signs = []
        cell_nodes_Q, cell_weigh_Q = [], []
        # ----------------------------------------------------------------------
        # For each face of the cell.
        # ----------------------------------------------------------------------
        for face, face_nodes in zip(faces, faces_nodes):
            # ------------------------------------------------------------------
            # Computing the distance from the cell barycenter to the face, and
            # the sign of the normal vector, depending on the location of the
            # cell barycenter to that of the face.
            # ------------------------------------------------------------------
            vector_to_face = self.get_vector_to_face(face, self.barycenter)
            if vector_to_face[-1] > 0:
                n_sign = 1.0
            else:
                n_sign = -1.0
            signs.append(n_sign)
            distance_to_face = np.abs(vector_to_face[-1])
            # ------------------------------------------------------------------
            # Computing the subcell quadrature points and faces.
            # ------------------------------------------------------------------
            # ##################################################################
            # 2D FACES IN 3D CELLS
            # ##################################################################
            cell_dimension = np.array([face.barycenter]).shape[1]
            if cell_dimension == 3:
                # --------------------------------------------------------------
                # Computing the volume of the pyramidal subcell with base that
                # of the face and height distance_to_face.
                # --------------------------------------------------------------
                sub_cell_volume = (1.0 / 3.0) * face.volume * distance_to_face
                if face_nodes.shape[0] > 4:
                    # ----------------------------------------------------------
                    # Initializing the vector to contain quadrature points and
                    # weights for each subcell.
                    # ----------------------------------------------------------
                    sub_cell_nodes_Q = []
                    sub_cell_weigh_Q = []
                    # ----------------------------------------------------------
                    # Creating subfaces segmentations.
                    # ----------------------------------------------------------
                    face_nodes_shifted = np.roll(face_nodes, 1, axis=0)
                    face_barycenter_matrix = np.tile(
                        face.barycenter, (face_nodes.shape[0], 1)
                    )
                    triangular_sub_faces = [
                        np.array(
                            [
                                face_nodes_shifted[i],
                                face_nodes[i],
                                face_barycenter_matrix[i],
                            ]
                        )
                        for i in range(face_nodes.shape[0])
                    ]
                    # ----------------------------------------------------------
                    # Initializing the vector to contain quadrature points and
                    # weights for each subface.
                    # ----------------------------------------------------------
                    for sub_face_nodes in triangular_sub_faces:
                        # ------------------------------------------------------
                        # Compute each subface triangle volume.
                        # ------------------------------------------------------
                        sub_face_volume = (
                            np.norm(sub_face_nodes[1] - sub_face_nodes[0])
                            * np.norm(sub_face_nodes[2] - sub_face_nodes[0])
                            / 2.0
                        )
                        sub_sub_cell_volume = distance_to_face * sub_face_volume
                        sub_sub_cell_nodes = np.concatenate(
                            (sub_face_nodes, [self.barycenter]), axis=0
                        )
                        # ------------------------------------------------------
                        # Compute quadrature nodes and weights.
                        # ------------------------------------------------------
                        (
                            sub_sub_cell_nodes_Q,
                            sub_sub_cell_weigh_Q,
                        ) = get_unite_tetrahedron_quadrature(
                            sub_sub_cell_nodes, sub_sub_cell_volume, k_Q
                        )
                        # ------------------------------------------------------
                        # Multiplying the quadrature weight with the face
                        # volume.
                        # ------------------------------------------------------
                        sub_sub_cell_weigh_Q = sub_sub_cell_volume * sub_sub_cell_weigh_Q
                        # ------------------------------------------------------
                        # Adding the subfaces quadrature nodes and weights to
                        # those of the global solution.
                        # ------------------------------------------------------
                        sub_cell_nodes_Q.append(sub_sub_cell_nodes_Q)
                        sub_cell_weigh_Q.append(sub_sub_cell_weigh_Q)
                    sub_cell_nodes_Q = np.concatenate(sub_cell_nodes_Q, axis=0)
                    sub_cell_weigh_Q = np.concatenate(sub_cell_weigh_Q, axis=0)
                elif face_nodes.shape[0] == 3:
                    # sub_face_volume = (
                    #     np.norm(face_nodes[1] - face_nodes[0])
                    #     * np.norm(face_nodes[2] - face_nodes[0])
                    #     / 2.0
                    # )
                    sub_cell_nodes = np.concatenate(
                        (face_nodes, [self.barycenter]), axis=0
                    )
                    # ----------------------------------------------------------
                    # Compute quadrature nodes and weights.
                    # ----------------------------------------------------------
                    (
                        sub_cell_nodes_Q,
                        sub_cell_weigh_Q,
                    ) = get_unite_tetrahedron_quadrature(
                        sub_cell_nodes, sub_cell_volume, k_Q
                    )
                    # ----------------------------------------------------------
                    # Multiplying the quadrature weight with the face
                    # volume.
                    # ----------------------------------------------------------
                    sub_cell_weigh_Q = sub_cell_volume * sub_cell_weigh_Q
            # ##################################################################
            # 1D FACES IN 2D CELLS
            # ##################################################################
            elif cell_dimension == 2:
                # --------------------------------------------------------------
                # Computing the volume of the pyramidal subcell with base that
                # of the face and height distance_to_face.
                # --------------------------------------------------------------
                sub_cell_volume = (1.0 / 2.0) * face.volume * distance_to_face
                sub_cell_nodes = np.concatenate((face_nodes, [self.barycenter]), axis=0)
                # --------------------------------------------------------------
                # Compute quadrature nodes and weights.
                # --------------------------------------------------------------
                (sub_cell_nodes_Q, sub_cell_weigh_Q,) = get_unite_triangle_quadrature(
                    sub_cell_nodes, sub_cell_volume, k_Q
                )
                # --------------------------------------------------------------
                # Multiplying the quadrature weight with the face
                # volume.
                # --------------------------------------------------------------
                sub_cell_weigh_Q = sub_cell_volume * sub_cell_weigh_Q
            # ##################################################################
            # 0D FACES IN 1D CELLS
            # ##################################################################
            elif cell_dimension == 1:
                # --------------------------------------------------------------
                # Computing the volume of the pyramidal subcell with base that
                # of the face and height distance_to_face.
                # --------------------------------------------------------------
                sub_cell_volume = (1.0 / 1.0) * face.volume * distance_to_face
                sub_cell_nodes = np.concatenate((face_nodes, [self.barycenter]), axis=0)
                # --------------------------------------------------------------
                # Compute quadrature nodes and weights.
                # --------------------------------------------------------------
                (sub_cell_nodes_Q, sub_cell_weigh_Q,) = get_unite_segment_quadrature(
                    sub_cell_nodes, sub_cell_volume, k_Q
                )
                # --------------------------------------------------------------
                # Multiplying the quadrature weight with the face
                # volume.
                # --------------------------------------------------------------
                sub_cell_weigh_Q = sub_cell_volume * sub_cell_weigh_Q
            # ##################################################################
            # 0D FACES IN 2D CELLS
            # ##################################################################
            cell_nodes_Q.append(sub_cell_nodes_Q)
            cell_weigh_Q.append(sub_cell_weigh_Q)
            cell_volume += sub_cell_volume
        # ----------------------------------------------------------------------
        # Concatenating quadrature points and weights for the whole cell
        # ----------------------------------------------------------------------
        signs = np.array(signs, dtype=int)
        # signs = create_vector(signs, dtype=int)
        cell_nodes_Q = np.concatenate(cell_nodes_Q, axis=0)
        cell_weigh_Q = np.concatenate(cell_weigh_Q, axis=0)
        return cell_volume, cell_nodes_Q, cell_weigh_Q, signs
