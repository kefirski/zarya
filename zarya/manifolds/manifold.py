class Manifold:
    def proj_(self, x, dim=-1):
        raise NotImplementedError

    def conf_factor(self, x=None, dim=-1, keepdim=False):
        raise NotImplementedError

    def add(self, x, y, dim=-1):
        raise NotImplementedError

    def mul(self, x, r, dim=-1):
        raise NotImplementedError

    def neg(self, x, dim=-1):
        return self.mul(x, -1, dim)

    def log(self, x, y, dim=-1):
        r"""
        Mapping of point y from Manifold to Tangent Space at point x
        """
        raise NotImplementedError

    def exp(self, x, v, dim=-1):
        r"""
        Mapping of point v from Tangent space at point x back to Manifold
        """
        raise NotImplementedError

    def zero_log(self, y, dim=-1):
        r"""
        Mapping of point y from Manifold to Tangent Space at point 0
        """
        raise NotImplementedError

    def zero_exp(self, v, dim=-1):
        r"""
        Mapping from Tangent space at point 0 of point v back to Manifold
        """
        raise NotImplementedError

    def parallel_transport(self, x, dim=-1, _from=None, _to=None):
        raise NotImplementedError

    def linear(self, x, m):
        r"""
        Fused composition of zero_log mapping, linear mapping and zero_exp mapping
        """
        raise NotImplementedError

    def hyperplane(self, x, p, a):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError
