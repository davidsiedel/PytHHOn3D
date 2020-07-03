import numpy as np
from numpy import ndarray as Mat
from scipy.special import binom

from bases.basis import Basis

# ----------------------------------------------------------------------------------------------------------------------
# One has the following matrices for the whole mesh:
# N, C_nc, C_nf, C_cf, weights, Nsets, flags
# ----------------------------------------------------------------------------------------------------------------------
class ScaledMonomial(Basis):
    def __init__(
        self, polynomial_order: int, domain_dimension: int,
    ):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The ScaledMonomial class provides a framework and methods to build a (scaled) polynomial
        basis given a polynomial order and a spatial dimension.
        The ScaledMonomial class is then used in other classes to compute matrices through
        quadrature rules.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - k : the polynomial order
        - d : the spatial dimension
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - dim : the dimension of the polynomial basis
        - pow_matrix : the exponent vector that acts on linear functions to build
        d-variate polynoms of order k in the polynomial basis
        - global_gradient_operator : the concatenation of gradients operators in all
        directions
        """
        super().__init__(polynomial_order, domain_dimension)
        # self.basis_dimension = int(binom(polynomial_order + domain_dimension, polynomial_order,))
        self.exponents = self.get_exponents(polynomial_order, domain_dimension,)
        self.gradient_operators = []
        for dx in range(domain_dimension):
            grad_dx = self.get_gradient_operator(polynomial_order, domain_dimension, dx)
            self.gradient_operators.append(grad_dx)

    def get_exponents(self, polynomial_order: int, domain_dimension: int) -> Mat:
        """
        ================================================================================================================
        Description :
        ================================================================================================================
        For a d-variate polynom of order k, computes the exponent vector to
        build the (scaled) monomial basis, where the exponent vector denotes the
        sum of mutli indexes alpha such that | alpha | < k.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - polynomial order
        - spatial dimension
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let consider the space of polynoms of :
        - order 1 in dimension 2 (I)
        - order 3 in dimension 1 (II)
        - order 1 in dimension 3 (III)
        the exponent vectors of such spaces are respectively :
        (I) | 0 0 |  (II) | 0 |  (III) | 0 0 0 |
            | 0 1 |       | 1 |        | 0 0 1 |
            | 1 0 |       | 2 |        | 0 1 0 |
            | 0 2 |       | 3 |        | 1 0 0 |
            | 1 1 |                             
            | 2 0 |                             
        and corresponding vectors of these bases are :
        (I) | x^0*y^0 |  (II) | x^0 |  (III) | x^0*y^0*z^0 |
            | x^0*y^1 |       | x^1 |        | x^0*y^0*z^1 |
            | x^1*y^0 |       | x^2 |        | x^0*y^1*z^0 |
            | x^0*y^2 |       | x^3 |        | x^1*y^0*z^0 |
            | x^1*y^1 |                               
            | x^2*y^0 |
        """
        if domain_dimension == 0:
            pow_matrix = np.array([[0.0]])
        else:
            pow_matrix = []
            if domain_dimension == 1:
                for s in range(polynomial_order + 1):
                    pow_matrix.append([s])
            elif domain_dimension == 2:
                for polynomial_order in range(polynomial_order + 1):
                    for l in range(polynomial_order + 1):
                        m = polynomial_order - l
                        pow_matrix.append(
                            [l, m,]
                        )
            elif domain_dimension == 3:
                for polynomial_order in range(polynomial_order + 1):
                    for l in range(polynomial_order + 1):
                        for m in range(polynomial_order + 1):
                            if not (l + m) > polynomial_order:
                                n = polynomial_order - (l + m)
                                pow_matrix.append(
                                    [l, m, n,]
                                )
        return np.array(pow_matrix, dtype=int,)

    def get_gradient_operator(self, polynomial_order: int, domain_dimension: int, dx: int) -> Mat:
        """
        ================================================================================================================
        Description :
        ================================================================================================================
        Computes the linear operator that acts on a vector in the monomial basis to return
        its derivative in the same basis.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - the polynomial order
        - the spatial dimension
        - the derivative direction
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let consider p(x) = a*x^0 + b*x^1 + c*x^2 in the monomial basis | 1, x, x^2 |
        of the space of polynomials of order 2 in dimension 1. Then :
        p'(x) = | 0 0 0 | @ | a | where | 0 0 0 | is the gradient operator.
                | 1 0 0 |   | b |       | 1 0 0 |
                | 0 2 0 |   | c |       | 0 2 0 |
        """
        # --------------------------------------------------------------------------------------------------------------
        # Initializing the conformal gradient operator with zeros
        # --------------------------------------------------------------------------------------------------------------
        gradient_operator = np.zeros((self.basis_dimension, self.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        # Copiying the exponent matrix
        # --------------------------------------------------------------------------------------------------------------
        d_pow_matrix = np.copy(self.exponents)
        # --------------------------------------------------------------------------------------------------------------
        # Deriving a polynom, hence substracting a one-filled vector at the
        # derivative variable position
        # --------------------------------------------------------------------------------------------------------------
        d_pow_matrix[:, dx,] = d_pow_matrix[:, dx,] - np.ones(d_pow_matrix[:, dx,].shape, dtype=int,)
        # --------------------------------------------------------------------------------------------------------------
        # Building the conformal gradient operator
        # --------------------------------------------------------------------------------------------------------------
        for i, coef in enumerate(self.exponents[:, dx,]):
            if not coef == 0:
                for j, exponents in enumerate(self.exponents):
                    if (exponents == d_pow_matrix[i]).all():
                        gradient_operator[i, j] = coef
                        break
        return gradient_operator

    def get_phi_vector(self, point: Mat, centroid: Mat, volume: float,) -> Mat:
        """
        ================================================================================================================
        Description :
        ================================================================================================================
        Computes the polynomial valued vector in the scaled monomial basis.
        shape be like : [ 1. -1.  0.  1. -0.  0.]
        ================================================================================================================
        Parameters :
        ================================================================================================================
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let the unite square T = [0,1]*[0,1] in R^2, A = (0.1, 0.4) a point in T, and
        B = (0.5, 0.5) the centroid of T. The volume of T is 1.
        Then, the polynomial valued vector of A in the scaled monomial basis of order 2
        in T is :
        phi_vector = | ((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^0 |
                     | ((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^1 |
                     | ((0.1-0.5)/1.)^1*((0.4-0.5)/1.)^0 |
                     | ((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^2 |
                     | ((0.1-0.5)/1.)^1*((0.4-0.5)/1.)^1 |
                     | ((0.1-0.5)/1.)^2*((0.4-0.5)/1.)^0 |
        """
        point_matrix = np.tile(point, (self.basis_dimension, 1,),)
        centroid_matrix = np.tile(centroid, (self.basis_dimension, 1,),)
        phi_matrix = ((point_matrix - centroid_matrix) / volume) ** self.exponents
        phi_vector = np.prod(phi_matrix, axis=1,)
        return phi_vector

    def get_d_phi_vector(self, point: Mat, centroid: Mat, volume: float, dx: int,) -> Mat:
        """
        ================================================================================================================
        Description :
        ================================================================================================================
        Computes the polynomial valued vector in the scaled monomial basis derivative with respect to a given direction.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let the unite square T = [0,1]*[0,1] in R^2, A = (0.1, 0.4) a point in T, and
        B = (0.5, 0.5) the centroid of T. The volume of T is 1.
        Then, the polynomial valued vector of A in the derivative of the scaled monomial
        basis of order 2 in T is :
        d_phi_vector = | 0*((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^0     |
                       | 0*((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^1     |
                       | 1*((0.1-0.5)/1.)^(1-1)*((0.4-0.5)/1.)^0 |
                       | 0*((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^2     |
                       | 1*((0.1-0.5)/1.)^(1-1)*((0.4-0.5)/1.)^1 |
                       | 2*((0.1-0.5)/1.)^(2-1)*((0.4-0.5)/1.)^0 |
        Or equivalently :
        d_phi_vector = (1/volume) @ gradient_operator @ phi_vector
        """
        grad_dx = self.gradient_operators[dx]
        phi_vector = self.get_phi_vector(point, centroid, volume)
        d_phi_vector = (1.0 / volume) * (grad_dx @ phi_vector.T)
        return d_phi_vector
