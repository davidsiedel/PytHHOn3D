import numpy as np


def get_point_quadrature():
    """
    Defining the point quadrature given a quadrature order k_Q
    Returns :
    - nodes_Q : quadrature points
    - weigh_Q : quadrature weights
    """
    nodes_Q, weigh_Q = np.array([[]]), np.array([[1.0]])
    return nodes_Q, weigh_Q


def get_unite_segment_quadrature(nodes, volume, k_Q):
    """
    Defining the unite trinagle quadrature given a quadrature order k_Q
    Returns :
    - nodes_Q : quadrature points
    - weigh_Q : quadrature weights
    """
    barycenter = 0.5000 * (nodes[0][0] + nodes[1][0])
    if k_Q == 1:
        nodes_Q = np.array([[barycenter]])
        weigh_Q = np.array([[volume]])
    if k_Q == 2:
        nodes_Q = np.array(
            [[barycenter - volume * 0.2887], [barycenter + volume * 0.2887],]
        )
        weigh_Q = np.array([[volume * 0.5000], [volume * 0.5000]])
    if k_Q == 3:
        nodes_Q = np.array(
            [
                [barycenter - volume * 0.3873],
                [barycenter],
                [barycenter + volume * 0.3873],
            ]
        )
        weigh_Q = np.array([[volume * 0.2778], [volume * 0.4444], [volume * 0.2778]])
    if k_Q == 4:
        nodes_Q = np.array(
            [
                [barycenter - volume * 0.4306],
                [barycenter + volume * 0.4306],
                [barycenter - volume * 0.1700],
                [barycenter + volume * 0.1700],
            ]
        )
        weigh_Q = np.array(
            [
                [volume * 0.1739],
                [volume * 0.1739],
                [volume * 0.3261],
                [volume * 0.3261],
            ]
        )
    return nodes_Q, weigh_Q


def get_unite_triangle_quadrature(nodes, volume, k_Q):
    """
    Defining the unite trinagle quadrature given a quadrature order k_Q
    Returns :
    - nodes_Q : quadrature points
    - weigh_Q : quadrature weights
    """
    if k_Q == 1:
        barycentric_coordinates = np.array([[0.3333, 0.3333, 0.3333]])
        nodes_Q = []
        for barycentric_coordinate in barycentric_coordinates:
            node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
            nodes_Q.append(node_Q)
        nodes_Q = np.array(nodes_Q)
        weigh_Q = np.array([[volume]])
    if k_Q == 2:
        barycentric_coordinates = np.array(
            [
                [0.6667, 0.1667, 0.1667],
                [0.1667, 0.6667, 0.1667],
                [0.1667, 0.1667, 0.6667],
            ]
        )
        nodes_Q = []
        for barycentric_coordinate in barycentric_coordinates:
            node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
            nodes_Q.append(node_Q)
        nodes_Q = np.array(nodes_Q)
        weigh_Q = np.array([[0.5000 * volume], [0.5000 * volume], [0.5000 * volume]])
    if k_Q == 3:
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
        weigh_Q = np.array(
            [
                [0.5625 * volume],
                [0.5208 * volume],
                [0.5208 * volume],
                [0.5208 * volume],
            ]
        )
    return nodes_Q, weigh_Q


def get_unite_tetrahedron_quadrature(nodes, volume, k_Q):
    """
    Defining the unite trinagle quadrature given a quadrature order k_Q
    Returns :
    - nodes_Q : quadrature points
    - weigh_Q : quadrature weights
    """
    if k_Q == 1:
        barycentric_coordinates = np.array([[0.2500, 0.2500, 0.2500, 0.2500]])
        nodes_Q = []
        for barycentric_coordinate in barycentric_coordinates:
            node_Q = np.sum((nodes * np.array([barycentric_coordinate]).T), axis=0)
            nodes_Q.append(node_Q)
        nodes_Q = np.array(nodes_Q)
        weigh_Q = np.array([[volume]])
    if k_Q == 2:
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
        weigh_Q = np.array(
            [
                [0.2500 * volume],
                [0.2500 * volume],
                [0.2500 * volume],
                [0.2500 * volume],
            ]
        )
    if k_Q == 3:
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
            [
                [-0.8000 * volume],
                [0.4500 * volume],
                [0.4500 * volume],
                [0.4500 * volume],
                [0.4500 * volume],
            ]
        )
    return nodes_Q, weigh_Q
