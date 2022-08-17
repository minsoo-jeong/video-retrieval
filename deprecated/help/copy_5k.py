from pathlib import Path
import shutil
import os
from tqdm import tqdm
import pickle as pk

with open('fivr_5k.txt', 'r') as f:
    fivr_5k = list(map(str.strip, f.readlines()))

print(fivr_5k)

src_root = Path('/mldisk/nfs_shared_/MLVD/FIVR/videos')
dst_root = Path('/mlsun/ms/fivr5k/videos')

d = dict()
for k in tqdm(fivr_5k):
    src = src_root.joinpath(k + '.mp4')
    if src.exists():
        dst = dst_root.joinpath(k + '.mp4')
        shutil.copy2(src.as_posix(), dst.as_posix())
        d[k] = dst.as_posix()

with open('/workspace/fivr5k_videos.pkl', 'wb') as f:
    pk.dump(d, f)

print(len(d.keys()))
