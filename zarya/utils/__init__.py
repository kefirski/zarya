from .check_view import _check_view
from .math import Atanh, Asinh

atanh = Atanh.apply
asinh = Asinh.apply

del Atanh
del Asinh
