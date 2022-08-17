import copy
import pickle

from utils.array import load_pickle
import torch

from datasets.datasets import FIVR
import numpy as np

from fractions import Fraction

index = np.array([1628])
clip_ids = torch.tensor([0])
print(clip_ids.dtype)

args = np.argsort(clip_ids)
print(args)
ss = index[args]
print(ss)


cc = np.take(index, args, axis=0)
print(cc)
sorted_index = np.take_along_axis(index, np.argsort(clip_ids), axis=0)
print(sorted_index)

exit()

a = np.array(['-0NokAd5WSY', '-0NokAd5WSY', '-0NokAd5WSY', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1nbjDwg4lc',
              '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-695tXnhqG4', '-695tXnhqG4', '-0NokAd5WSY',
              '-0NokAd5WSY', '-0NokAd5WSY', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1nbjDwg4lc', '-1nbjDwg4lc',
              '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-695tXnhqG4', '-695tXnhqG4', '-0NokAd5WSY', '-0NokAd5WSY',
              '-0NokAd5WSY', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc',
              '-1nbjDwg4lc', '-1nbjDwg4lc', '-695tXnhqG4', '-695tXnhqG4', '-0NokAd5WSY', '-0NokAd5WSY', '-0NokAd5WSY',
              '-1SnlEMzOi4', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc',
              '-1nbjDwg4lc', '-695tXnhqG4', '-695tXnhqG4', '-0NokAd5WSY', '-0NokAd5WSY', '-0NokAd5WSY', '-1SnlEMzOi4',
              '-1SnlEMzOi4', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc',
              '-695tXnhqG4', '-0NokAd5WSY', '-0NokAd5WSY', '-0NokAd5WSY', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1SnlEMzOi4',
              '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-695tXnhqG4',
              '-0NokAd5WSY', '-0NokAd5WSY', '-0NokAd5WSY', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1nbjDwg4lc',
              '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-695tXnhqG4', '-0NokAd5WSY',
              '-0NokAd5WSY', '-0NokAd5WSY', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1SnlEMzOi4', '-1nbjDwg4lc', '-1nbjDwg4lc',
              '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-1nbjDwg4lc', '-695tXnhqG4', '-0NokAd5WSY'])
b = np.array(['-0NokAd5WSY', '-1SnlEMzOi4', '-1nbjDwg4lc', '-695tXnhqG4']).reshape(-1, 1)
print(b)

w = [np.where(a == bb) for bb in b]

print(w)

# print([np.where(a==bb) for bb in b])
# print([np.argwhere(a==bb) for bb in b])

exit()

exit()

a = {'a': 123, 'b': 456, 'c': [1, 2, 3]}
b = copy.deepcopy(a)

b.pop('c')
print(a)
print(b)

exit()

a = Fraction(0.6)
print(a)
print(round(a))
exit()
a = np.random.choice([1, 2, 3, 4, 5], 1, replace=False)
print(a)

exit()
fivr = FIVR('5k')

print(fivr.get_all_keys())
print(len(fivr.get_all_keys()))

exit()

a = load_pickle('/workspace/fivr_frames.pkl')
print(a)
exit()

a = torch.cuda.is_available()
print(a)
b = torch.cuda.device_count()
print(b)

exit()
a = list(range(10))
print(a)

print(a[1:3])

a = load_pickle('/workspace/fivr_frames.pickle')
print(a.keys())
print(a)
exit()
b = dict()
for k, v in a.items():
    vv = v.replace('hdd', 'mlsun')
    b[k] = vv

# pickle.dump(b, open('/workspace/vcdb_resnet50-avgpool-mlsun.pkl','wb'))
