import numpy as np
import os
import tqdm

p = '/mlsun/ms/vcdb/resnet50-avgpool'
l = os.listdir(p)

ll = [i for i in l if len(i) > len('yWVmXRx3mFg.npy')]
print(l)
print(len(ll))

f = []
for i in tqdm.tqdm(ll):
    f.append(np.load(os.path.join(p, i)))

print(len(f))

ff = np.concatenate(f)
print(ff.shape)
