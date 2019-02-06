class Manifold:
    def proj_(self, x):
        raise NotImplementedError

    def add(self, x, y, dim):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError
