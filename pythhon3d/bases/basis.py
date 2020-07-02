from scipy.special import binom


class Basis:
    def __init__(
        self, polynomial_order: int, domain_dimension: int,
    ):
        self.basis_dimension = int(binom(polynomial_order + domain_dimension, polynomial_order,))
