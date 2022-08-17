import json
import os
from tqdm import tqdm

with open('../fivr5k_info.json', 'r') as f:
    index = json.load(f)

for k, v in tqdm(index.items()):
    nb_frame = v['nb_frames']
    frame_dir = v['frames']
    decoded = os.listdir(frame_dir)
    if nb_frame != len(decoded):
        imgs = [i for i in decoded if i.endswith('.jpg')]

        print(k, v, len(decoded), decoded)
