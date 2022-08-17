import pickle as pk

from pathlib import Path

import pickle as pk

from utils.array import load_pickle
from tqdm import tqdm
import os

# fivr = load_pickle('/workspace/fivr_frames.pickle')
# vcdb = load_pickle('/workspace/vcdb_frames_mlsun.pickle')
# print(fivr)

# for k, v in tqdm(vcdb.items()):
#     if not os.path.exists(v):
#         print(k, v)
#
# for k, v in tqdm(fivr.items()):
#     if not os.path.exists(v):
#         print(k, v)

vcdb_root = Path('/mlsun/ms/vcdb/frames/')
vcdb = {p.stem: p.as_posix() for p in vcdb_root.rglob('*.npy')}
with open('/workspace/vcdb_frames.pkl', 'wb') as f:
    pk.dump(vcdb, f)
print(len(vcdb.keys()))

fivr_root = Path('/mldisk/nfs_shared_/ms/fivr/frames/')
fivr = {p.stem: p.as_posix() for p in fivr_root.glob('*.npy')}
with open('/workspace/fivr_frames.pkl', 'wb') as f:
    pk.dump(fivr, f)
print(len(fivr.keys()))
