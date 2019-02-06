from functools import reduce
from operator import mul


def check_view(old_size, new_size, hdim):
    left_prod = reduce(mul, old_size[:hdim], 1)
    right_prod = reduce(mul, old_size[hdim + 1 :], 1)

    left = mul_until_eq(new_size, left_prod)
    right = len(new_size) - 1 - mul_until_eq(reversed(new_size), right_prod)

    result = (
        left >= 0
        and right >= 0
        and right - left == 2
        and new_size[left + 1] == old_size[hdim]
    )
    return result, left + 1 if result else -1


def mul_until_eq(arr, val):
    acc = 1
    for i, item in enumerate(arr):
        acc *= item
        if acc == val:
            return i

    return -1
