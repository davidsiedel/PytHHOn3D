from pythhon3d.core.domain import Domain
import numpy as np
import pytest

test_domain_barycenter_vector_data = []
test_simplicial_volume_data = []

# ========================================================================================
# DIMENSION 0
# ========================================================================================
# ...................... INPUT -> Vertices
# --------------------------------------------------------------------
domain_vertices_matrix = np.array([[0.0]])
# ...................... OUTPUT : Barycenter
# --------------------------------------------------------------------
expected_domain_barycenter_vector = np.array([[0.0]])
test_domain_barycenter_vector_data.append(
    (domain_vertices_matrix, expected_domain_barycenter_vector)
)
# ...................... OUTPUT : Simplex volume
# --------------------------------------------------------------------
expecetd_simplicial_volume = 1.0
test_simplicial_volume_data.append((domain_vertices_matrix, expecetd_simplicial_volume))

# ========================================================================================
# DIMENSION 1
# ========================================================================================
# ...................... INPUT -> Vertices
# --------------------------------------------------------------------
domain_vertices_matrix = np.array([[0.0], [1.0]])
# ...................... OUTPUT : Barycenter
# --------------------------------------------------------------------
expected_domain_barycenter_vector = np.array([[0.5]])
test_domain_barycenter_vector_data.append(
    (domain_vertices_matrix, expected_domain_barycenter_vector)
)
# ...................... OUTPUT : Simplex volume
# --------------------------------------------------------------------
expecetd_simplicial_volume = 1.0
test_simplicial_volume_data.append((domain_vertices_matrix, expecetd_simplicial_volume))

# ========================================================================================
# DIMENSION 2
# ========================================================================================
# ...................... INPUT -> Vertices
# --------------------------------------------------------------------
domain_vertices_matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
# ...................... OUTPUT : Barycenter
# --------------------------------------------------------------------
expected_domain_barycenter_vector = np.array([1.0 / 3.0, 1.0 / 3.0])
test_domain_barycenter_vector_data.append(
    (domain_vertices_matrix, expected_domain_barycenter_vector)
)
# ...................... OUTPUT : Simplex volume
# --------------------------------------------------------------------
expecetd_simplicial_volume = 0.5
test_simplicial_volume_data.append((domain_vertices_matrix, expecetd_simplicial_volume))

# ========================================================================================
# DIMENSION 3
# ========================================================================================
# ...................... INPUT -> Vertices
# --------------------------------------------------------------------
domain_vertices_matrix = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
# ...................... OUTPUT : Barycenter
# --------------------------------------------------------------------
expected_domain_barycenter_vector = np.array([0.25, 0.25, 0.25])
test_domain_barycenter_vector_data.append(
    (domain_vertices_matrix, expected_domain_barycenter_vector)
)
# ...................... OUTPUT : Simplex volume
# --------------------------------------------------------------------
expecetd_simplicial_volume = 1.0 / 6.0
test_simplicial_volume_data.append((domain_vertices_matrix, expecetd_simplicial_volume))


@pytest.mark.parametrize(
    "domain_vertices_matrix, expected_domain_barycenter_vector",
    test_domain_barycenter_vector_data,
)
def test_get_domain_barycenter_vector(
    domain_vertices_matrix, expected_domain_barycenter_vector
):
    domain = Domain(domain_vertices_matrix)
    domain_barycenter_vector = domain.get_domain_barycenter_vector(domain_vertices_matrix)
    assert (
        np.abs(domain_barycenter_vector - expected_domain_barycenter_vector)
        < np.full(expected_domain_barycenter_vector.shape, 1.0e-5)
    ).all()


@pytest.mark.parametrize(
    "domain_vertices_matrix, expecetd_simplicial_volume", test_simplicial_volume_data,
)
def test_get_simplicial_volume(domain_vertices_matrix, expecetd_simplicial_volume):
    domain = Domain(domain_vertices_matrix)
    domain_edges_matrix = domain.get_domain_edges_matrix(domain_vertices_matrix)
    simplicial_volume = domain.get_simplicial_volume(domain_edges_matrix)
    assert (np.abs(simplicial_volume - expecetd_simplicial_volume) < 1.0e-5).all()
