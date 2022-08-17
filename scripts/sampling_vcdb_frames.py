import numpy as np
import json
import os
from tqdm import tqdm
from datasets.utils import load_index


def sampling_n_frames_per_videos(index, frames_per_video=10, format='{:06d}.jpg'):
    index = load_index(index)
    frames = []
    pbar = tqdm(index.items())
    for k, v in pbar:
        frame_dir = v.get('frames')
        if frame_dir:
            size = min(frames_per_video, v['nb_frames'])
            indices = np.random.choice(v['nb_frames'], size=size, replace=False)
            paths = list(map(lambda x: os.path.join(frame_dir, format.format(x + 1)), indices))

            frames.extend(paths)
            pbar.set_description(f'{len(frames)} frames')
    pbar.close()

    return frames


if __name__ == '__main__':
    np.random.seed(42)

    frames = sampling_n_frames_per_videos('../data/vcdb-90k.json', 10)
    with open('../data/vcdb-90k-10frames.json', 'w') as f:
        json.dump(frames, f, indent=2)
