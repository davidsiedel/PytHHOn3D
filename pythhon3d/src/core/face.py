import numpy as np
from numpy import ndarray as Mat
from HHO.Quadrature import *
from HHO.Boundary import Boundary
from typing import List

# ------------------------------------------------------------------------------
# One has the following matrices for the whole mesh:
# N, C_nc, C_nf, C_cf, weights, Nsets, flags
# ------------------------------------------------------------------------------
class Face:
    def __init__(self, face_nodes: Mat, k_Q: int, boundary: Boundary = None):
        """
        kjbljbj
        """
        self.boundary = boundary
        self.barycenter = self.get_face_barycenter(face_nodes)
        self.p_matrix = self.get_face_local_reference_frame_transformation_matrix(
            face_nodes, self.barycenter
        )
        self.normal_vector = self.get_face_normal(self.p_matrix)
        self.volume, self.nodes_Q, self.weigh_Q = self.get_face_integration_data(
            self.p_matrix, face_nodes, self.barycenter, k_Q
        )
        self.diameter = np.sqrt((4.0 * self.volume) / np.pi)
        return

    def get_face_barycenter(self, nodes: Mat) -> Mat:
        """
        Getting the barycenter of any d dimensional spatial domain given its nodes
        Returns :
        - barycenter : the face barycenter
        """
        barycenter = [np.mean(nodes[:, i]) for i in range(nodes.shape[1])]
        return np.array(barycenter)

    def get_face_local_reference_frame_transformation_matrix(
        self, face_nodes: Mat, face_barycenter: Mat
    ) -> Mat:
        """
        Getting the reference frame transformation matrix of a d-1 dimensional
        spatial domain.
        Returns :
        - p_matrix : the reference frame transformation matrix
        """
        # ######################################################################
        # 2D FACES IN 3D CELLS
        # ######################################################################
        if face_nodes.shape[1] == 3:
            # e_0 = face_barycenter - face_nodes[0, :]
            e_0 = face_nodes[0] - face_barycenter
            # e_0 = face_nodes[1] - face_nodes[0]
            e_0 = e_0 / np.linalg.norm(e_0)
            # e_test = face_barycenter - face_nodes[1, :]
            e_test = face_nodes[1] - face_barycenter
            # e_test = face_nodes[2] - face_nodes[0]
            e_2 = np.cross(e_0, e_test)
            e_2 = e_2 / np.linalg.norm(e_2)
            e_1 = np.cross(e_2, e_0)
            # e_1 = np.cross(e_0, e_2)
            p_matrix = np.array([e_0, e_1, e_2])
        # ######################################################################
        # 1D FACES IN 2D CELLS
        # ######################################################################
        elif face_nodes.shape[1] == 2:
            e_0 = face_nodes[1, :] - face_nodes[0, :]
            e_1 = np.array([e_0[1], -e_0[0]])
            p_matrix = np.array([e_0, e_1])
        # ######################################################################
        # 0D FACES IN 1D CELLS
        # ######################################################################
        elif face_nodes.shape[1] == 1:
            p_matrix = np.array([[1.0]])
        return p_matrix

    def get_face_normal(self, p_matrix):
        """
        Returns the normal vector to a face :
        """
        return p_matrix[-1]

    def get_face_height(self, p_matrix, face_barycenter):
        """
        Returns the normal vector to a face :
        """
        return ((p_matrix @ face_barycenter.T).T)[-1]

    def get_face_integration_data(self, p_matrix, face_nodes, face_barycenter, k_Q):
        """
        Returns the normal vector to a face :
        """
        # ######################################################################
        # 2D FACES IN 3D CELLS
        # ######################################################################
        if face_nodes.shape[1] == 3:
            # ------------------------------------------------------------------
            # Expressing the face nodes and barycenter in the local reference
            # frame.
            # ------------------------------------------------------------------
            face_nodes_loc = (p_matrix @ face_nodes.T).T
            face_barycenter_loc = (p_matrix @ face_barycenter.T).T
            face_height_loc = face_nodes_loc[0, -1]
            if face_nodes.shape[0] > 3:
                # --------------------------------------------------------------
                # Initializing the volume scalar for the whole face.
                # --------------------------------------------------------------
                face_volume = 0.0
                # --------------------------------------------------------------
                # Creating subfaces segmentations.
                # --------------------------------------------------------------
                face_nodes_shifted = np.roll(face_nodes_loc, 1, axis=0)
                face_barycenter_matrix = np.tile(
                    face_barycenter_loc, (face_nodes_loc.shape[0], 1)
                )
                triangular_sub_faces = [
                    np.array(
                        [
                            face_nodes_shifted[i, :-1],
                            face_nodes_loc[i, :-1],
                            face_barycenter_matrix[i, :-1],
                        ]
                    )
                    for i in range(face_nodes_loc.shape[0])
                ]
                # --------------------------------------------------------------
                # Initializing the vector to contain quadrature points and
                # weights for each subface.
                # --------------------------------------------------------------
                face_nodes_Q, face_weigh_Q = [], []
                for sub_face_nodes in triangular_sub_faces:
                    # ----------------------------------------------------------
                    # Compute each subface triangle volume.
                    # ----------------------------------------------------------
                    edge_0 = np.linalg.norm(sub_face_nodes[1] - sub_face_nodes[0])
                    edge_1 = np.linalg.norm(sub_face_nodes[0] - sub_face_nodes[2])
                    edge_2 = np.linalg.norm(sub_face_nodes[2] - sub_face_nodes[1])
                    s = (edge_0 + edge_1 + edge_2) / 2.0
                    sub_face_volume = (
                        s * (s - edge_0) * (s - edge_1) * (s - edge_2)
                    ) ** 0.5
                    face_volume += sub_face_volume
                    # ----------------------------------------------------------
                    # Compute each subface quadrature nodes and weights.
                    # ----------------------------------------------------------
                    (
                        sub_face_nodes_Q,
                        sub_face_weigh_Q,
                    ) = get_unite_triangle_quadrature(
                        sub_face_nodes, sub_face_volume, k_Q
                    )
                    # ----------------------------------------------------------
                    # Multiplying the quadrature weights for each suvface with
                    # their volume.
                    # ----------------------------------------------------------
                    sub_face_weigh_Q = sub_face_volume * sub_face_weigh_Q
                    # ----------------------------------------------------------
                    # Adding the subfaces quadrature nodes and weights to those
                    # of the global solution.
                    # ----------------------------------------------------------
                    face_nodes_Q.append(sub_face_nodes_Q)
                    face_weigh_Q.append(sub_face_weigh_Q)
                face_nodes_Q = np.concatenate(face_nodes_Q, axis=0)
                face_weigh_Q = np.concatenate(face_weigh_Q, axis=0)
            elif face_nodes.shape[0] == 3:
                # --------------------------------------------------------------
                # Compute the face volume.
                # --------------------------------------------------------------
                edge_0 = np.linalg.norm(face_nodes[1] - face_nodes[0])
                edge_1 = np.linalg.norm(face_nodes[0] - face_nodes[2])
                edge_2 = np.linalg.norm(face_nodes[2] - face_nodes[1])
                s = (edge_0 + edge_1 + edge_2) / 2.0
                face_volume = (s * (s - edge_0) * (s - edge_1) * (s - edge_2)) ** 0.5
                # --------------------------------------------------------------
                # Compute each subface quadrature nodes and weights.
                # --------------------------------------------------------------
                (face_nodes_Q, face_weigh_Q,) = get_unite_triangle_quadrature(
                    face_nodes_loc[:, :-1], face_volume, k_Q
                )
                # --------------------------------------------------------------
                # Multiplying the quadrature weights for each suvface with
                # their volume.
                # --------------------------------------------------------------
                face_weigh_Q = face_volume * face_weigh_Q
            # ------------------------------------------------------------------
            # Adding the face height to the quadratue nodes, that were computed
            # in a hyperplane, so that the face integral is expressed within
            # the cell dimension.
            # ------------------------------------------------------------------
            h_vector = np.full((face_nodes_Q.shape[0], 1), face_height_loc)
            face_nodes_Q = np.concatenate((face_nodes_Q, h_vector), axis=1)
        # ######################################################################
        # 1D FACES IN 2D CELLS
        # ######################################################################
        elif face_nodes.shape[1] == 2:
            # ------------------------------------------------------------------
            # Expressing the nodes and barycenter in the local reference frame.
            # ------------------------------------------------------------------
            face_nodes_loc = (p_matrix @ face_nodes.T).T
            face_height_loc = face_nodes_loc[0, -1]
            # ------------------------------------------------------------------
            # Computing the segment vector.
            # ------------------------------------------------------------------
            face_volume = (face_nodes_loc[1, :-1][0] - face_nodes_loc[0, :-1][0]) / 2.0
            # ------------------------------------------------------------------
            # Compute quadrature nodes and weights.
            # ------------------------------------------------------------------
            face_nodes_Q, face_weigh_Q = get_unite_segment_quadrature(
                face_nodes_loc[:, :-1], face_volume, k_Q
            )
            # ------------------------------------------------------------------
            # Multiplying the quadrature weight with the face volume.
            # ------------------------------------------------------------------
            face_weigh_Q = face_volume * face_weigh_Q
            # ------------------------------------------------------------------
            # Adding the face height to the quadratue nodes, that were computed
            # in a hyperplane, so that the face integral is expressed within
            # the cell dimension.
            # ------------------------------------------------------------------
            h_vector = np.full((face_nodes_Q.shape[0], 1), face_height_loc)
            face_nodes_Q = np.concatenate((face_nodes_Q, h_vector), axis=1)
        # ######################################################################
        # 0D FACES IN 1D CELLS
        # ######################################################################
        elif face_nodes.shape[1] == 1:
            face_volume = 1.0
            face_nodes_loc = (p_matrix @ face_nodes.T).T
            face_height_loc = face_nodes_loc[0, -1]
            # ------------------------------------------------------------------
            # Compute quadrature nodes and weights.
            # ------------------------------------------------------------------
            face_nodes_Q, face_weigh_Q = get_point_quadrature()
            # ------------------------------------------------------------------
            # Multiplying the quadrature weight with the face volume.
            # ------------------------------------------------------------------
            face_weigh_Q = face_volume * face_weigh_Q
            # ------------------------------------------------------------------
            # Adding the face height to the quadratue nodes, that were computed
            # in a hyperplane, so that the face integral is expressed within
            # the cell dimension.
            # ------------------------------------------------------------------
            h_vector = np.full((face_nodes_Q.shape[0], 1), face_height_loc)
            face_nodes_Q = np.concatenate((face_nodes_Q, h_vector), axis=1)
        # ######################################################################
        # EXPRESSING THE QUADRATURE NODES IN THE NATURAL COORDINATE SYSTEM
        # ######################################################################
        face_nodes_Q = (p_matrix.T @ face_nodes_Q.T).T
        return face_volume, face_nodes_Q, face_weigh_Q
