import numpy as np


class Quadrature:
    def __init__(self, rule: str = "DUNAVANT"):
        """
        ==================================================================================
        Class :
        ==================================================================================
        The Quadrature class provides inytegration methods.
        ==================================================================================
        Parameters :
        ==================================================================================
        ==================================================================================
        Attributes :
        ==================================================================================
        """
        self.rule = rule

    @staticmethod
    def get_unite_point_quadrature():
        """
        Defining the point quadrature given a quadrature order k
        Returns :
        - nodes_Q : quadrature points
        - weigh_Q : quadrature weights
        """
        nodes_Q, weigh_Q = np.array([[]]), np.array([[1.0]])
        return nodes_Q, weigh_Q

    @staticmethod
    def get_unite_segment_quadrature(nodes, volume, k):
        """
        Defining the unite trinagle quadrature given a quadrature order k
        Returns :
        - nodes_Q : quadrature points
        - weigh_Q : quadrature weights
        """
        barycenter = 0.5000 * (nodes[0][0] + nodes[1][0])
        if k == 1:
            nodes_Q = np.array([[barycenter]])
            weigh_Q = np.array([[volume]])
        if k in [2, 3]:
            nodes_Q = np.array(
                [[barycenter - volume * 0.28867513459481287], [barycenter + volume * 0.28867513459481287],]
            )
            weigh_Q = np.array([[volume * 0.5000], [volume * 0.5000]])
        if k in [4, 5]:
            nodes_Q = np.array(
                [[barycenter - volume * 0.3872983346207417], [barycenter], [barycenter + volume * 0.3872983346207417],]
            )
            weigh_Q = np.array(
                [[volume * 0.2777777777777778], [volume * 0.4444444444444444], [volume * 0.2777777777777778],]
            )
        return nodes_Q, weigh_Q

    @staticmethod
    def get_unite_triangle_quadrature(nodes, volume, k):
        """
        Defining the unite trinagle quadrature given a quadrature order k
        Returns :
        - nodes_Q : quadrature points
        - weigh_Q : quadrature weights
        """
        if k == 1:
            barycentric_coordinates = np.array([[0.3333, 0.3333, 0.3333]])
            nodes_Q = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
                nodes_Q.append(node_Q)
            nodes_Q = np.array(nodes_Q)
            weigh_Q = np.array([[volume]])
        if k == 2:
            barycentric_coordinates = np.array(
                [[0.6667, 0.1667, 0.1667], [0.1667, 0.6667, 0.1667], [0.1667, 0.1667, 0.6667],]
            )
            nodes_Q = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
                nodes_Q.append(node_Q)
            nodes_Q = np.array(nodes_Q)
            weigh_Q = np.array([[0.3333 * volume], [0.3333 * volume], [0.3333 * volume]])
        if k == 3:
            barycentric_coordinates = np.array(
                [
                    [0.3333, 0.3333, 0.3333],
                    [0.6000, 0.2000, 0.2000],
                    [0.2000, 0.6000, 0.2000],
                    [0.2000, 0.2000, 0.6000],
                ]
            )
            nodes_Q = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
                nodes_Q.append(node_Q)
            nodes_Q = np.array(nodes_Q)
            weigh_Q = np.array([[0.5625 * volume], [0.5208 * volume], [0.5208 * volume], [0.5208 * volume],])
        return nodes_Q, weigh_Q

    @staticmethod
    def get_unite_tetrahedron_quadrature(nodes, volume, k):
        """
        Defining the unite trinagle quadrature given a quadrature order k
        Returns :
        - nodes_Q : quadrature points
        - weigh_Q : quadrature weights
        """
        if k == 1:
            barycentric_coordinates = np.array([[0.2500, 0.2500, 0.2500, 0.2500]])
            nodes_Q = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
                nodes_Q.append(node_Q)
            nodes_Q = np.array(nodes_Q)
            weigh_Q = np.array([[volume]])
        if k == 2:
            barycentric_coordinates = np.array(
                [
                    [0.5854, 0.1382, 0.1382, 0.1382],
                    [0.1382, 0.5854, 0.1382, 0.1382],
                    [0.1382, 0.1382, 0.5854, 0.1382],
                    [0.1382, 0.1382, 0.1382, 0.5854],
                ]
            )
            nodes_Q = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
                nodes_Q.append(node_Q)
            nodes_Q = np.array(nodes_Q)
            weigh_Q = np.array([[0.2500 * volume], [0.2500 * volume], [0.2500 * volume], [0.2500 * volume],])
        if k == 3:
            barycentric_coordinates = np.array(
                [
                    [0.2500, 0.2500, 0.2500, 0.2500],
                    [0.5000, 0.1667, 0.1667, 0.1667],
                    [0.1667, 0.5000, 0.1667, 0.1667],
                    [0.1667, 0.1667, 0.5000, 0.1667],
                    [0.1667, 0.1667, 0.1667, 0.5000],
                ]
            )
            nodes_Q = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
                nodes_Q.append(node_Q)
            nodes_Q = np.array(nodes_Q)
            weigh_Q = np.array(
                [[-0.8000 * volume], [0.4500 * volume], [0.4500 * volume], [0.4500 * volume], [0.4500 * volume],]
            )
        return nodes_Q, weigh_Q
