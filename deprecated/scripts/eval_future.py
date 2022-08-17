import einops
import numpy as np
import torch

from utils.array import load_json
from scripts.extract_x3d_ddp import FrameClipsDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import models_3d
import os
import random
from pathlib import Path
from tqdm import tqdm

# from datasets.datasets import FIVR


from futures.datasets import FIVR
from futures.models import Average


def load_features(root_dir):
    features = dict()
    for p in tqdm(Path(root_dir).rglob('*.npy')):
        features[p.stem] = np.load(p.as_posix())
    return features


if __name__ == '__main__':
    features = load_features('/mlsun/ms/fivr5k/slow_r50_8_8_224')
    print(len(features.keys()))

    fivr = FIVR('5k', annotation='/workspace/datasets/fivr.tca.pickle')

    average = Average()
    with torch.no_grad():
        video_features = {k: average(torch.tensor(v).unsqueeze(0),
                                     torch.tensor([v.shape[0]])).numpy()
                          for k, v in tqdm(features.items())
                          }

    print(fivr.evaluate(video_features, metric='cosine'))

    clip_features = features
    print(fivr.evaluate(clip_features, metric='chamfer'))
