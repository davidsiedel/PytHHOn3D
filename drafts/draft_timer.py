# class Exemple:
#     def __init__(self, a):
#         self.a = a


# class Chose(Exemple):
#     def __init__(self, a):
#         super().__init__(a)


# class Truc(Exemple, Chose):
#     def __init__(self, a):
#         Chose.__init__(self, a)


# a = 2
# truc = Truc(a)
import numpy as np
import time


def get_domain_shape(vertices) -> str:
    number_of_vertices = vertices.shape[0]
    problem_dimension = vertices.shape[1]
    if number_of_vertices == 1 and problem_dimension == 1:
        shape = "POINT"
    if number_of_vertices == 2 and problem_dimension == 1:
        shape = "SEGMENT"
    if number_of_vertices == 3 and problem_dimension == 2:
        shape = "TRIANGLE"
    if number_of_vertices == 4 and problem_dimension == 2:
        shape = "QUADRANGLE"
    if number_of_vertices > 4 and problem_dimension == 2:
        shape = "POLYGON"
    if number_of_vertices == 4 and problem_dimension == 3:
        shape = "TETRAHEDRON"
    if number_of_vertices == 8 and problem_dimension == 3:
        shape = "HEXAHEDRON"
    if number_of_vertices > 8 and problem_dimension == 3:
        shape = "POLYHEDRON"
    return shape


start_time = time.time()
for i in range(2000000):
    # print(i)
    mat = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
        ]
    )
    get_domain_shape(mat)
print("--- %s seconds ---" % (time.time() - start_time))
