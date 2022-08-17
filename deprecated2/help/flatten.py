import itertools
from itertools import chain

a = [[1, 2], [3, 4, 5], [6, 7], [8, [9, 10]]]

b = list(itertools.chain(*a))
print(b)

a = [{1, 2}, {3, 4, 5}, {6, 7}, {1, 2, 3}]
b = set(itertools.chain(*a))
print(b)
