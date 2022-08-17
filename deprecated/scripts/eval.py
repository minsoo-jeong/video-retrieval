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
from datasets.datasets import FIVR
import torch.nn.functional as F
from utils.measure import calculate_ap


def chamfer_distance(a, b):
    sim = cosine_distance(a, b)
    chamfer = sim.max(dim=1)[0].sum() / sim.shape[0]
    return chamfer


def cosine_distance(a, b):
    at = torch.tensor(a)
    bt = torch.tensor(b)
    an = F.normalize(at, p=2, dim=1)
    bn = F.normalize(bt, p=2, dim=1)
    return torch.einsum('ik,jk->ij', [an, bn])


def average(f):
    f = torch.tensor(f)
    f = F.normalize(f, p=2, dim=1)
    f = einops.reduce(f, 'i k -> 1 k', 'mean')
    f = F.normalize(f, p=2, dim=1)

    return f


def load_features(root_dir):
    features = dict()
    for p in tqdm(Path(root_dir).rglob('*.npy')):
        features[p.stem] = np.load(p.as_posix())
    return features


if __name__ == '__main__':
    features = load_features('/mlsun/ms/fivr5k/slow_r50_8_8_224')
    print(features.keys())

    fivr = FIVR('5k',pickle='/workspace/datasets/fivr.tca.pickle')
    print(fivr)

    print(fivr.evaluate({k: average(v) for k, v in features.items()}))

    queries = fivr.get_queries()
    reference = fivr.get_database()
    print(len(queries))
    print(reference)

    query_features = np.concatenate([features[q] for q in queries])
    reference_features = np.concatenate([features[q] for q in reference])

    dsvr, csvr, isvr = 0., 0., 0.
    for n, q in enumerate(queries):
        fq = features[q]
        dist = []
        for m, r in enumerate(reference):
            fr = features[r]
            dist.append(chamfer_distance(fq, fr))
        idx = np.argsort(dist)[::-1]
        annotation = fivr.get_annotation(q)

        gt = annotation.get('ND', [])
        gt += annotation.get('DS', [])
        gt_idx = {reference.index(g) for g in gt}
        dsvr += calculate_ap(idx, gt_idx)

        gt += annotation.get('CS', [])
        gt_idx = {reference.index(g) for g in gt}
        csvr += calculate_ap(idx, gt_idx)

        gt += annotation.get('IS', [])
        gt_idx = {reference.index(g) for g in gt}
        isvr += calculate_ap(idx, gt_idx)

    dsvr /= len(queries)
    csvr /= len(queries)
    isvr /= len(queries)

    print(dsvr, csvr, isvr)
