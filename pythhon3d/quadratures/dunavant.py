import numpy as np


class DunavantRule:
    def __init__(self):
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

    @staticmethod
    def get_point_quadrature():
        """
        Defining the point quadrature given a quadrature order k
        Returns :
        - quadrature_nodes : quadrature points
        - quadrature_weights : quadrature weights
        """
        quadrature_nodes, quadrature_weights = np.array([[]]), np.array([[1.0]])
        return quadrature_nodes, quadrature_weights

    @staticmethod
    def get_segment_quadrature(vertices, volume, k):
        """
        Defining the unite trinagle quadrature given a quadrature order k
        Returns :
        - quadrature_nodes : quadrature points
        - quadrature_weights : quadrature weights
        """
        barycenter = 0.5000 * (vertices[0][0] + vertices[1][0])
        if k == 1:
            quadrature_nodes = np.array([[barycenter]])
            quadrature_weights = np.array([[volume]])
        if k in [2, 3]:
            quadrature_nodes = np.array(
                [[barycenter - volume * 0.28867513459481287], [barycenter + volume * 0.28867513459481287],]
            )
            quadrature_weights = np.array([[volume * 0.5000], [volume * 0.5000]])
        if k in [4, 5]:
            quadrature_nodes = np.array(
                [[barycenter - volume * 0.3872983346207417], [barycenter], [barycenter + volume * 0.3872983346207417],]
            )
            quadrature_weights = np.array(
                [[volume * 0.2777777777777778], [volume * 0.4444444444444444], [volume * 0.2777777777777778],]
            )
        return quadrature_nodes, quadrature_weights

    @staticmethod
    def get_triangle_quadrature(vertices, volume, k):
        """
        Defining the unite trinagle quadrature given a quadrature order k
        Returns :
        - quadrature_nodes : quadrature points
        - quadrature_weights : quadrature weights
        """
        if k == 1:
            barycentric_coordinates = np.array([[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]])
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array([[volume]])
        if k == 2:
            barycentric_coordinates = np.array(
                [
                    [0.66666666666666666, 0.16666666666666666, 0.16666666666666666],
                    [0.16666666666666666, 0.66666666666666666, 0.16666666666666666],
                    [0.16666666666666666, 0.16666666666666666, 0.66666666666666666],
                ]
            )
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array(
                [[0.3333333333333333 * volume], [0.3333333333333333 * volume], [0.3333333333333333 * volume]]
            )
        if k == 3:
            barycentric_coordinates = np.array(
                [
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                    [0.6000000000000000, 0.2000000000000000, 0.2000000000000000],
                    [0.2000000000000000, 0.6000000000000000, 0.2000000000000000],
                    [0.2000000000000000, 0.2000000000000000, 0.6000000000000000],
                ]
            )
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array(
                [
                    [-0.5625000000000000 * volume],
                    [0.5208333333333333 * volume],
                    [0.5208333333333333 * volume],
                    [0.5208333333333333 * volume],
                ]
            )
        if k == 4:
            barycentric_coordinates = np.array(
                [
                    [0.8168475729804590, 0.0915762135097710, 0.0915762135097710],
                    [0.0915762135097710, 0.8168475729804590, 0.0915762135097710],
                    [0.0915762135097710, 0.0915762135097710, 0.8168475729804590],
                    [0.1081030181680700, 0.4459484909159650, 0.4459484909159650],
                    [0.4459484909159650, 0.1081030181680700, 0.4459484909159650],
                    [0.4459484909159650, 0.4459484909159650, 0.1081030181680700],
                ]
            )
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array(
                [
                    [0.109951743655322 * volume],
                    [0.109951743655322 * volume],
                    [0.109951743655322 * volume],
                    [0.223381589678011 * volume],
                    [0.223381589678011 * volume],
                    [0.223381589678011 * volume],
                ]
            )
        if k == 5:
            barycentric_coordinates = np.array(
                [
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                    [0.7974269853530870, 0.1012865073234560, 0.1012865073234560],
                    [0.1012865073234560, 0.7974269853530870, 0.1012865073234560],
                    [0.1012865073234560, 0.1012865073234560, 0.7974269853530870],
                    [0.0597158717897700, 0.4701420641051150, 0.4701420641051150],
                    [0.4701420641051150, 0.0597158717897700, 0.4701420641051150],
                    [0.4701420641051150, 0.4701420641051150, 0.0597158717897700],
                ]
            )
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array(
                [
                    [0.225000000000000 * volume],
                    [0.125939180544827 * volume],
                    [0.125939180544827 * volume],
                    [0.125939180544827 * volume],
                    [0.132394152788506 * volume],
                    [0.132394152788506 * volume],
                    [0.132394152788506 * volume],
                ]
            )
        return quadrature_nodes, quadrature_weights

    @staticmethod
    def get_tetrahedron_quadrature(vertices, volume, k):
        """
        Defining the unite trinagle quadrature given a quadrature order k
        Returns :
        - quadrature_nodes : quadrature points
        - quadrature_weights : quadrature weights
        """
        if k == 1:
            barycentric_coordinates = np.array(
                [[0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000]]
            )
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array([[volume]])
        if k == 2:
            barycentric_coordinates = np.array(
                [
                    [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
                    [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
                    [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
                    [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
                ]
            )
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array(
                [
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                ]
            )
        if k == 3:
            barycentric_coordinates = np.array(
                [
                    [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
                    [0.5000000000000000, 0.1666666666666667, 0.1666666666666667, 0.1666666666666667],
                    [0.1666666666666667, 0.5000000000000000, 0.1666666666666667, 0.1666666666666667],
                    [0.1666666666666667, 0.1666666666666667, 0.5000000000000000, 0.1666666666666667],
                    [0.1666666666666667, 0.1666666666666667, 0.1666666666666667, 0.5000000000000000],
                ]
            )
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array(
                [
                    [-0.8000000000000000 * volume],
                    [0.4500000000000000 * volume],
                    [0.4500000000000000 * volume],
                    [0.4500000000000000 * volume],
                    [0.4500000000000000 * volume],
                ]
            )
        # if k == 4:
        #     barycentric_coordinates = np.array(
        #         [
        #             [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
        #             [0.7857142857142860, 0.0714285714285710, 0.0714285714285710, 0.0714285714285710],
        #             [0.0714285714285710, 0.7857142857142860, 0.0714285714285710, 0.0714285714285710],
        #             [0.0714285714285710, 0.0714285714285710, 0.7857142857142860, 0.0714285714285710],
        #             [0.0714285714285710, 0.0714285714285710, 0.0714285714285710, 0.7857142857142860],
        #             #
        #             [0.3994035761667990, 0.3994035761667990, 0.1005964238332010, 0.1005964238332010],
        #             [0.3994035761667990, 0.2500000000000000, 0.3994035761667990, 0.2500000000000000],
        #             [0.3994035761667990, 0.2500000000000000, 0.2500000000000000, 0.3994035761667990],
        #             [0.2500000000000000, 0.3994035761667990, 0.3994035761667990, 0.2500000000000000],
        #             [0.2500000000000000, 0.3994035761667990, 0.2500000000000000, 0.3994035761667990],
        #             [0.2500000000000000, 0.2500000000000000, 0.3994035761667990, 0.3994035761667990],
        #         ]
        #     )
        #     quadrature_nodes = []
        #     for barycentric_coordinate in barycentric_coordinates:
        #         node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
        #         quadrature_nodes.append(node_Q)
        #     quadrature_nodes = np.array(quadrature_nodes)
        #     quadrature_weights = np.array(
        #         [
        #             [-0.0131555555555556 * volume],
        #             [0.0076222222222222 * volume],
        #             [0.0076222222222222 * volume],
        #             [0.0076222222222222 * volume],
        #             [0.0076222222222222 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #         ]
        #     )
        if k in [4, 5]:
            barycentric_coordinates = np.array(
                [
                    [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
                    #
                    [0.8160578438945539, 0.0919710780527230, 0.0919710780527230, 0.0919710780527230],
                    [0.0919710780527230, 0.8160578438945539, 0.0919710780527230, 0.0919710780527230],
                    [0.0919710780527230, 0.0919710780527230, 0.8160578438945539, 0.0919710780527230],
                    [0.0919710780527230, 0.0919710780527230, 0.0919710780527230, 0.8160578438945539],
                    #
                    [0.3604127443407402, 0.3197936278296299, 0.3197936278296299, 0.3197936278296299],
                    [0.3197936278296299, 0.3604127443407402, 0.3197936278296299, 0.3197936278296299],
                    [0.3197936278296299, 0.3197936278296299, 0.3604127443407402, 0.3197936278296299],
                    [0.3197936278296299, 0.3197936278296299, 0.3197936278296299, 0.3604127443407402],
                    #
                    [0.1381966011250105, 0.1381966011250105, 0.3618033988749895, 0.3618033988749895],
                    [0.1381966011250105, 0.3618033988749895, 0.1381966011250105, 0.3618033988749895],
                    [0.1381966011250105, 0.3618033988749895, 0.3618033988749895, 0.1381966011250105],
                    [0.3618033988749895, 0.1381966011250105, 0.1381966011250105, 0.3618033988749895],
                    [0.3618033988749895, 0.1381966011250105, 0.3618033988749895, 0.1381966011250105],
                    [0.3618033988749895, 0.3618033988749895, 0.1381966011250105, 0.1381966011250105],
                ]
            )
            quadrature_nodes = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_nodes.append(node_Q)
            quadrature_nodes = np.array(quadrature_nodes)
            quadrature_weights = np.array(
                [
                    [0.1185185185185185 * volume],
                    [0.0719370837790186 * volume],
                    [0.0719370837790186 * volume],
                    [0.0719370837790186 * volume],
                    [0.0719370837790186 * volume],
                    [0.0690682072262724 * volume],
                    [0.0690682072262724 * volume],
                    [0.0690682072262724 * volume],
                    [0.0690682072262724 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                ]
            )
        return quadrature_nodes, quadrature_weights
