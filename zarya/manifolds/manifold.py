class Manifold:
    def proj_(self, x):
        raise NotImplementedError

    def conf_factor(self, x, dim, keepdim=False):
        raise NotImplementedError

    def add(self, x, y, dim):
        raise NotImplementedError

    def mul(self, x, r, dim):
        raise NotImplementedError

    def neg(self, x, dim):
        return self.mul(x, -1, dim)

    def log(self, x, y, dim):
        r"""
        Mapping of point y from Manifold to Tangent Space at point x
        """
        raise NotImplementedError

    def exp(self, x, v, dim):
        r"""
        Mapping from Tangent space at point x of point v back to Manifold
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError
