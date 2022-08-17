import pickle
from pathlib import Path
import pickle as pk
import os

import random
import torchvision.datasets.video_utils
from torch.utils.data import Dataset, DataLoader
# from utils import decode
from pathlib import PosixPath

from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch

import numpy as np
from PIL import Image

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data.dataloader import default_collate

import utils.array
from utils.array import similarity_search, load_pickle, resize_array
from utils.measure import calculate_ap

from itertools import chain
from typing import Union

FIVR_PKL = Path(__file__).parent.joinpath('fivr.dns.pickle')
VCDB_PICKLE = Path(__file__).parent.joinpath('vcdb.pkl')


class FIVR(object):
    def __init__(self, version, pickle=FIVR_PKL):
        self.version = version
        with open(pickle, 'rb') as f:
            dataset = pk.load(f)

        self.queries = dataset[self.version]['queries']
        self.database = list(dataset[self.version]['database'])
        self.annotation = self.rebuild_annotation(dataset['annotation'])

    def get_database(self):
        return sorted(self.database)

    def get_queries(self):
        return sorted(self.queries)

    def rebuild_annotation(self, o):
        db = self.get_database()
        query = self.get_queries()
        annotation = dict()
        for q in query:
            annotation[q] = {k: list({a for a in ann if a in db and a != q}) for k, ann in o[q].items()}
        return annotation

    def get_all_keys(self):
        return sorted(list(set(self.database + self.queries)))

    def get_annotation(self, query=None):
        ann = self.annotation
        if query is not None:
            ann = ann.get(query)
        return ann

    def evaluate(self, features):
        # features(dict)
        queries, databases = self.get_queries(), self.get_database()

        query_features = np.concatenate([features[q] for q in queries])
        database_features = np.concatenate([features[q] for q in databases])
        dist, idx = similarity_search(query_features, database_features)

        dsvr, csvr, isvr = 0., 0., 0.
        for n, q in enumerate(queries):
            annotation = self.get_annotation(q)
            gt = annotation.get('ND', [])
            gt += annotation.get('DS', [])
            dsvr += calculate_ap(idx[n], {databases.index(g) for g in gt})
            gt += annotation.get('CS', [])
            csvr += calculate_ap(idx[n], {databases.index(g) for g in gt})
            gt += annotation.get('IS', [])
            isvr += calculate_ap(idx[n], {databases.index(g) for g in gt})
        dsvr /= len(queries)
        csvr /= len(queries)
        isvr /= len(queries)
        return dsvr, csvr, isvr


class VCDB(object):
    def __init__(self, pickle=VCDB_PICKLE):
        dataset = utils.array.load_pickle(pickle)

        self.pair = [tuple(p['videos']) for p in dataset['pair']]
        self.distraction = sorted(dataset['distraction'])

    def sampling_negative(self, count=1):
        return np.random.choice(self.distraction, count, replace=False)


class ListDataset(Dataset):
    def __init__(self, items, transform=None):
        super(ListDataset, self).__init__()
        self.items = items
        self.transform = transform

    def __getitem__(self, idx):
        item = self.items[idx]

        if self.transform is not None:
            item = self.transform(image=np.array(item))['image']

        return item

    def __len__(self):
        return len(self.items)


TRNF_ResizedCenterCrop = A.Compose([A.SmallestMaxSize(max_size=256),
                                    A.CenterCrop(height=224, width=224),
                                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    ToTensorV2()
                                    ])

TRNF_Default = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                          ToTensorV2()])

default_transform = A.Compose([
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()])


class VCDBTripletDataset(Dataset):
    # anc - pos : pair
    # negative : distraction
    def __init__(self, index, n_negative=1, length=64, transform=None):
        self.vcdb = VCDB()
        self.index = load_pickle(index)
        self.n_negative = n_negative
        self.length = length
        self.transform = transform

    def __len__(self):
        return len(self.vcdb.pair)

    def __getitem__(self, idx):
        pair = self.vcdb.pair[idx]
        negative = self.vcdb.sampling_negative(self.n_negative)
        anc_path, pos_path = self.index[pair[0]], self.index[pair[1]]
        neg_paths = [self.index[n] for n in negative]

        anc, len_a = load_numpy_array(anc_path, self.length, self.transform)
        pos, len_p = load_numpy_array(pos_path, self.length, self.transform)
        neg, len_n = zip(*[load_numpy_array(n, self.length, self.transform) for n in neg_paths])
        neg = torch.stack(neg)

        return anc, pos, neg, len_a, len_p, len_n


class VCDBDistractionDataset(Dataset):
    def __init__(self, index, length=64, transform=None):
        self.vcdb = VCDB()
        self.index = load_pickle(index)
        self.length = length
        self.transform = transform

    def __len__(self):
        return len(self.vcdb.distraction)

    def __getitem__(self, idx):
        key = self.vcdb.distraction[idx]
        path = self.index[key]

        frames, len_f = load_numpy_array(path, self.length, self.transform)

        return key, frames, len_f


def VCDBTripletDataset_collate_fn(batch):
    anc, pos, neg, len_a, len_p, len_n = zip(*batch)

    return default_collate(anc), default_collate(pos), torch.cat(neg, dim=0), \
           default_collate(len_a), default_collate(len_p), default_collate(list(chain(*len_n)))


def apply_transform(x, transform):
    if isinstance(transform, A.Compose):
        x = [transform(image=_x)['image'] for _x in x]
    else:
        x = [transform(_x) for _x in x]
    return torch.stack(x, dim=0)


def load_numpy_array(path, length=None, transform=None):
    # np.array path -> torch.tensor

    x = np.load(path)
    if np.any(np.isnan(x)):
        print('np.load nan', path, x.shape)

    size = x.shape[0]
    if length is not None and length != size:
        x, size = resize_array(x, length)
        if np.any(np.isnan(x)):
            print('resize_array nan', path, x.shape)

    if transform is not None:
        x = apply_transform(x, transform)
    else:
        x = torch.tensor(x)

    if torch.any(torch.isnan(x)):
        print('transform nan', path, x.shape)

    return x, size


class FIVRDataset(Dataset):
    def __init__(self, index, version, length=None, transform=None):
        super(FIVRDataset, self).__init__()
        self.fivr = FIVR(version)
        self.index = load_pickle(index)
        self.keys = self.fivr.get_all_keys()
        self.length = length
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        path = self.index[key]

        x, size = load_numpy_array(path, self.length, self.transform)

        return key, x, size


class FIVRFrameDataset(Dataset):
    def __init__(self, index: dict, version: str, length: int = None, transform: Union[None, A.ReplayCompose] = None):
        super(FIVRFrameDataset, self).__init__()
        self.fivr = FIVR(version)
        self.index = load_pickle(index)
        self.keys = self.fivr.get_all_keys()
        self.length = length
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        path = self.index[key]

        x = np.load(path)

        if self.length is not None:
            x, length = resize_array(x, x.shape[0])

        if self.transform is not None:
            pass
        else:
            x = torch.tensor(x)

        x, size = load_numpy_array(path, self.length, self.transform)

        return key, x, size


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

    # dataset = VCDBTripletDataset('/workspace/vcdb_resnet50-avgpool.pickle', n_negative=16, length=20)
    dataset = VCDBDistractionDataset('/workspace/vcdb_frames_mlsun.pickle', length=32)
    loader = DataLoader(dataset, batch_size=16, num_workers=8)
    for k, f, len_f in loader:
        print(f.shape, len_f, k)
    exit()
