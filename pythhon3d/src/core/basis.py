import numpy as np
from numpy import ndarray as Mat
from scipy.special import binom

# --------------------------------------------------------------------------------------
# One has the following matrices for the whole mesh:
# N, C_nc, C_nf, C_cf, weights, Nsets, flags
# --------------------------------------------------------------------------------------
class Basis:
    def __init__(
        self, k: int, d: int,
    ):
        self.dim = int(binom(k + d, k,))
        self.pow_matrix = self.get_pow_matrix(k, d,)

    def get_pow_matrix(self, k: int, d: int,) -> Mat:
        """
        For a d-variate polynom of order k, computes the exponent vector to
        build the scaled monomial basis monomial. The exponent vector is the
        sum of mutli indexes alpha such that | alpha | < k
        - pow_matrix
        """
        if d == 0:
            pow_matrix = np.array([[0.0]])
        else:
            pow_matrix = []
            if d == 1:
                for s in range(k + 1):
                    pow_matrix.append([s])
            elif d == 2:
                for k in range(k + 1):
                    for l in range(k + 1):
                        m = k - l
                        pow_matrix.append(
                            [l, m,]
                        )
            elif d == 3:
                for k in range(k + 1):
                    for l in range(k + 1):
                        for m in range(k + 1):
                            if not (l + m) > k:
                                n = k - (l + m)
                                pow_matrix.append(
                                    [l, m, n,]
                                )
        return np.array(pow_matrix, dtype=int,)

    def get_phi_vector(
        self, point: Mat, barycenter: Mat, volume: float,
    ) -> Mat:
        """
        hebakjbvkjbkbkhjvb kh
        """
        point_matrix = np.tile(point, (self.pow_matrix.shape[0], 1,),)
        barycenter_matrix = np.tile(barycenter, (self.pow_matrix.shape[0], 1,),)
        phi_matrix = (
            (point_matrix - barycenter_matrix) / volume
        ) ** self.pow_matrix
        phi_vector = np.prod(phi_matrix, axis=1,)
        return phi_vector

    def get_d_phi_vector(
        self, point: Mat, barycenter: Mat, volume: float, dx: int,
    ) -> Mat:
        """
        ce ke c-a fzait
        """
        # ----------------------------------------------------------------------
        # Copiying the exponent matrix
        # ----------------------------------------------------------------------
        d_pow_matrix = np.copy(self.pow_matrix)
        # ----------------------------------------------------------------------
        # Deriving a polynom, hence substracting a one-filled vector at the
        # derivative variable position
        # ----------------------------------------------------------------------
        d_pow_matrix[:, dx,] = d_pow_matrix[:, dx,] - np.ones(
            d_pow_matrix[:, dx,].shape, dtype=int,
        )
        # ----------------------------------------------------------------------
        # Taking the absolute value of the derivative exponent matrix :
        # Deriving constants (X**0) -> (0*X**(-1)) yields a divide by zero
        # operation. By taking the aboslute value of the derivative exponent
        # matrix, the -1 exponent becomes 1 but the value of the evaluation
        # of the derivative at X remains unchanged because of the
        # multiplication by 0.
        # ----------------------------------------------------------------------
        d_pow_matrix = np.abs(d_pow_matrix)
        # ----------------------------------------------------------------------
        # Creating a point vector with size that of the exponent matrix
        # ----------------------------------------------------------------------
        point_matrix = np.tile(point, (self.pow_matrix.shape[0], 1,),)
        barycenter_matrix = np.tile(barycenter, (self.pow_matrix.shape[0], 1,),)
        # ----------------------------------------------------------------------
        # Creating a point vector with size that of the exponent matrix
        # ----------------------------------------------------------------------
        point_matrix[:, dx,] = (
            self.pow_matrix[:, dx,] * point_matrix[:, dx,]
        )
        d_phi_matrix = (
            (point_matrix - barycenter_matrix) / volume
        ) ** d_pow_matrix
        d_phi_vector = np.prod(d_phi_matrix, axis=1,)

        return d_phi_vector
