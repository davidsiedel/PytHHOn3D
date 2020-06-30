from core.domain import Domain
import numpy as np

# ========================================================================================
# DIMENSION 0
# ========================================================================================
print("DIMENSION : 0")
# ...................... INPUT -> vertices_matrix
# ------------------------------------------------------------------------------
vertices_matrix = np.array([[0.0]])
d = Domain(vertices_matrix)
# ...................... OUTPUT : domain_barycenter_vector
# ------------------------------------------------------------------------------
domain_barycenter_vector = d.get_domain_barycenter_vector(vertices_matrix)
print("domain_barycenter_vector : \n {}".format(domain_barycenter_vector))
# ...................... OUTPUT : domain_edges_matrix
# ------------------------------------------------------------------------------
domain_edges_matrix = d.get_domain_edges_matrix(vertices_matrix)
print("domain_edges_matrix : \n {}".format(domain_edges_matrix))
# ...................... OUTPUT : simplicial_volume
# ------------------------------------------------------------------------------
simplicial_volume = d.get_simplicial_volume(domain_edges_matrix)
print("simplicial_volume : \n {}".format(simplicial_volume))

# ========================================================================================
# DIMENSION 1
# ========================================================================================
print("DIMENSION : 1")
# ...................... INPUT -> vertices_matrix
# ------------------------------------------------------------------------------
vertices_matrix = np.array([[0.0], [1.0]])
d = Domain(vertices_matrix)
# ...................... OUTPUT : domain_barycenter_vector
# ------------------------------------------------------------------------------
domain_barycenter_vector = d.get_domain_barycenter_vector(vertices_matrix)
print("domain_barycenter_vector : \n {}".format(domain_barycenter_vector))
# ...................... OUTPUT : domain_edges_matrix
# ------------------------------------------------------------------------------
domain_edges_matrix = d.get_domain_edges_matrix(vertices_matrix)
print("domain_edges_matrix : \n {}".format(domain_edges_matrix))
# ...................... OUTPUT : simplicial_volume
# ------------------------------------------------------------------------------
simplicial_volume = d.get_simplicial_volume(domain_edges_matrix)
print("simplicial_volume : \n {}".format(simplicial_volume))

# ========================================================================================
# DIMENSION 2
# ========================================================================================
print("DIMENSION : 2")
# ...................... INPUT -> vertices_matrix
# ------------------------------------------------------------------------------
vertices_matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
d = Domain(vertices_matrix)
# ...................... OUTPUT : domain_barycenter_vector
# ------------------------------------------------------------------------------
domain_barycenter_vector = d.get_domain_barycenter_vector(vertices_matrix)
print("domain_barycenter_vector : \n {}".format(domain_barycenter_vector))
# ...................... OUTPUT : domain_edges_matrix
# ------------------------------------------------------------------------------
domain_edges_matrix = d.get_domain_edges_matrix(vertices_matrix)
print("domain_edges_matrix : \n {}".format(domain_edges_matrix))
# ...................... OUTPUT : simplicial_volume
# ------------------------------------------------------------------------------
simplicial_volume = d.get_simplicial_volume(domain_edges_matrix)
print("simplicial_volume : \n {}".format(simplicial_volume))

# ========================================================================================
# DIMENSION 3
# ========================================================================================
print("DIMENSION : 3")
# ...................... INPUT -> vertices_matrix
# ------------------------------------------------------------------------------
vertices_matrix = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
d = Domain(vertices_matrix)
# ...................... OUTPUT : domain_barycenter_vector
# ------------------------------------------------------------------------------
domain_barycenter_vector = d.get_domain_barycenter_vector(vertices_matrix)
print("domain_barycenter_vector : \n {}".format(domain_barycenter_vector))
# ...................... OUTPUT : domain_edges_matrix
# ------------------------------------------------------------------------------
domain_edges_matrix = d.get_domain_edges_matrix(vertices_matrix)
print("domain_edges_matrix : \n {}".format(domain_edges_matrix))
# ...................... OUTPUT : simplicial_volume
# ------------------------------------------------------------------------------
simplicial_volume = d.get_simplicial_volume(domain_edges_matrix)
print("simplicial_volume : \n {}".format(simplicial_volume))
