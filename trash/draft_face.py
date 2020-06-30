from core.face import Face
from core.quadrature import Quadrature
from core.boundary import Boundary
import numpy as np


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

k = 1

# 1D
fig = plt.figure()
ax = fig.add_subplot(111)
face_vertices_matrix = np.array([[0.4]])
face = Face(face_vertices_matrix, k)
print("face_volume : \n{}\n".format(face.volume))
print("face_quadrature_points : \n{}\n".format(face.quadrature_points))
print("face_quadrature_weights : \n{}\n".format(face.quadrature_weights))
plt.show()
plt.close()

# 2D
fig = plt.figure()
ax = fig.add_subplot(111)
face_vertices_matrix = np.array([[0.4, 2.0], [0.8, 0.1]])
face_vertices_matrix = np.array([[1.0, 1.0], [1.0, 2.0]])
face = Face(face_vertices_matrix, k)
print("face_volume : \n{}\n".format(face.volume))
print("face_quadrature_points : \n{}\n".format(face.quadrature_points))
print("face_quadrature_weights : \n{}\n".format(face.quadrature_weights))
ax.scatter(face_vertices_matrix[:, 0], face_vertices_matrix[:, 1], c="r", marker="o")
ax.scatter(face.quadrature_points[:, 0], face.quadrature_points[:, 1], c="b", marker="^")
plt.show()
plt.close()
# 3D

print("========================")
fig = plt.figure()
ax = fig.add_subplot(111)
face_vertices_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
# face_vertices_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
face = Face(face_vertices_matrix, k)
print("face_volume : \n{}\n".format(face.volume))
print("face_quadrature_points : \n{}\n".format(face.quadrature_points))
print("face_quadrature_weights : \n{}\n".format(face.quadrature_weights))
ax.scatter(face_vertices_matrix[:, 0], face_vertices_matrix[:, 1], c="r", marker="o")
ax.scatter(face.quadrature_points[:, 0], face.quadrature_points[:, 1], c="b", marker="^")

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
plt.show()
