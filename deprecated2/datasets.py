import random

from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import pil_loader,default_loader
import torch
import einops
import numpy as np

from fractions import Fraction
from pathlib import Path
from typing import Union
import pickle as pk
import json
import os
import itertools
from pytorchvideo.data.encoded_video import EncodedVideo
from futures.distance import *
import math
from albumentations.pytorch import ToTensorV2
import albumentations as A

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_pickle(path):
    with open(path, 'rb') as f:
        content = pk.load(f)
    return content


def load_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content


def load_index(index):
    if isinstance(index, Path):
        index = index.as_posix()

    if isinstance(index, str):
        _, ext = os.path.splitext(index)
        if ext.lower() == '.json':
            index = load_json(index)
        elif ext.lower() in ['.pkl', '.pickle']:
            index = load_pickle(index)
        else:
            raise NotImplementedError(f'Unsupported file extension ({index})')

    elif isinstance(index, dict):
        index = index
    else:
        raise NotImplementedError(f'Unsupported type index ({index.__class__()})')

    return index


class VideoDataset(Dataset):
    def __init__(self, index: Union[dict, str, Path], frames_per_clip, sampling_rate, sampling_unit='frames',
                 n_clips=None, transform=None):
        self.index = load_index(index)
        self.keys = sorted(self.index.keys())
        self.frames_per_clip = frames_per_clip
        self.sampling_rate = sampling_rate
        self.sampling_unit = sampling_unit.lower()
        assert self.sampling_unit in ['frames', 'fps']
        self.transform = transform
        self.FRAME_NAME_FORMAT = '{:06d}.jpg'
        self.n_clips = n_clips  # number of clips per video

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        info = self.index[key]

        frames, n_clips = self.load_clips(info, self.frames_per_clip, self.n_clips)

        if self.transform:
            frames = self.apply_transform(frames, self.transform)
        clips = einops.rearrange(frames, '(n t) c h w -> n c t h w', t=self.frames_per_clip)

        return key, clips, n_clips

    def load_images(self, paths):
        return [pil_loader(path) for path in paths]

    def apply_transform(self, frames, transform):

        aug = transform(image=np.asarray(frames[0]))['replay']
        frames = [transform.replay(aug, image=np.asarray(f))['image'] for f in frames]

        frames = torch.stack(frames, dim=0)

        return frames

    def get_sampling_rate(self, info):
        if self.sampling_unit == 'fps':
            fps = Fraction(info.get('average_rate'))
            sampling_rate = round(Fraction(fps, self.sampling_rate))
        else:
            sampling_rate = self.sampling_rate

        return sampling_rate

    def load_clips(self, info, frames_per_clip=1, n_clip=None):

        sampling_rate = self.get_sampling_rate(info)

        frame_dir = info.get('frames')
        if frame_dir:
            # load from directory
            clips, indices = self.load_clips_from_frames(info,
                                                         sampling_rate=sampling_rate,
                                                         frames_per_clip=frames_per_clip,
                                                         n_clip=n_clip)

            indices = np.clip(indices, 0, info['nb_frames'] - 1)

            paths = list(map(lambda x: os.path.join(frame_dir, self.FRAME_NAME_FORMAT.format(x + 1)), indices))

            frames = self.load_images(paths)

        else:
            # Decode video
            clips, start_sec, end_sec, indices = self.load_clips_from_video(info,
                                                                            sampling_rate=sampling_rate,
                                                                            frames_per_clip=frames_per_clip,
                                                                            n_clip=n_clip)

            video = EncodedVideo.from_path(info['video'], decode_audio=False, decoder='pyav')
            frames = video.get_clip(start_sec, end_sec + 1e-6)['video']

            indices = torch.tensor(np.clip(indices, 0, frames.shape[1] - 1))
            frames = torch.index_select(frames, dim=1, index=indices)
            frames = einops.rearrange(frames, 'c t h w -> t h w c')

        return frames, clips

    def load_clips_from_frames(self, info, sampling_rate=30, frames_per_clip=1, n_clip=None):

        len_clip = frames_per_clip * sampling_rate
        clips = max(round(info['nb_frames'] / len_clip), 1)

        offset, n_frames = 0, clips * frames_per_clip

        if n_clip:
            n_frames = n_clip * frames_per_clip
            clips = min(clips, n_clip)
            if info['nb_frames'] > len_clip * clips:
                offset = np.random.choice(info['nb_frames'] - len_clip * clips)  # random offset

        indices = np.arange(0, n_frames) * sampling_rate + offset

        return clips, indices

    def load_clips_from_video(self, info, sampling_rate=30, frames_per_clip=1, n_clip=None):
        len_clip = frames_per_clip * sampling_rate
        fps = Fraction(info['average_rate'])
        nb_frames = math.floor(info['duration'] * fps)
        clips = max(round(nb_frames / len_clip), 1)

        offset, n_frames = 0, clips * frames_per_clip
        if n_clip:
            n_frames = n_clip * frames_per_clip
            clips = min(clips, n_clip)
            if nb_frames > len_clip * clips:
                offset = np.random.choice(nb_frames - len_clip * clips)

        start_sec = Fraction(offset, fps)
        end_sec = start_sec + Fraction(clips * len_clip, fps)

        indices = np.arange(0, n_frames) * sampling_rate

        return clips, start_sec, end_sec, indices

    @staticmethod
    def collate(batch):
        keys, frames, nb_clips = tuple(zip(*batch))
        frames = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
        return default_collate(keys), frames, default_collate(nb_clips)


class VideoDataset_VCDBPair(VideoDataset):
    def __init__(self, vcdb, *args, **kwargs):
        super(VideoDataset_VCDBPair, self).__init__(*args, **kwargs)

        self.pair = [p for p in vcdb.pair if p[0] in self.index.keys() and p[1] in self.index.keys()]

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        anc, pos = self.pair[idx]
        info_a = self.index[anc]
        info_p = self.index[pos]

        frames_a, n_clips_a = self.load_clips(info_a, self.frames_per_clip, self.n_clips)
        frames_p, n_clips_p = self.load_clips(info_p, self.frames_per_clip, self.n_clips)

        frames_a = self.apply_transform(frames_a, self.transform)
        frames_p = self.apply_transform(frames_p, self.transform)

        clips_a = einops.rearrange(frames_a, '(n t) c h w -> n c t h w', t=self.frames_per_clip)
        clips_p = einops.rearrange(frames_p, '(n t) c h w -> n c t h w', t=self.frames_per_clip)

        return anc, pos, clips_a, clips_p, n_clips_a, n_clips_p

    @staticmethod
    def collate(batch):
        anc, pos, clips_a, clips_p, n_clips_a, n_clips_p = tuple(zip(*batch))
        clips_a = torch.nn.utils.rnn.pad_sequence(clips_a, batch_first=True)
        clips_p = torch.nn.utils.rnn.pad_sequence(clips_p, batch_first=True)
        return (default_collate(anc),
                default_collate(pos),
                clips_a,
                clips_p,
                default_collate(n_clips_a),
                default_collate(n_clips_p))


class VideoDataset_VCDBTriplet(VideoDataset):
    def __init__(self, vcdb, negative=1, *args, **kwargs):
        super(VideoDataset_VCDBTriplet, self).__init__(*args, **kwargs)

        self.pair = [p for p in vcdb.pair if p[0] in self.index.keys() and p[1] in self.index.keys()]
        self.distractions = [v for v in vcdb.distraction if v in self.index.keys()]
        self.negative = negative

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        anc, pos = self.pair[idx]
        neg = np.random.choice(self.distractions, self.negative, replace=False)
        info_a = self.index[anc]
        info_p = self.index[pos]

        frames_a, n_clips_a = self.load_clips(info_a, self.frames_per_clip, self.n_clips)
        frames_p, n_clips_p = self.load_clips(info_p, self.frames_per_clip, self.n_clips)

        frames_n, n_clips_n = zip(*[self.load_clips(self.index[n], self.frames_per_clip, self.n_clips) for n in neg])

        frames_a = self.apply_transform(frames_a, self.transform)
        frames_p = self.apply_transform(frames_p, self.transform)
        frames_n = torch.stack([self.apply_transform(n, self.transform) for n in frames_n])

        clips_a = einops.rearrange(frames_a, '(n t) c h w -> n c t h w', t=self.frames_per_clip)
        clips_p = einops.rearrange(frames_p, '(n t) c h w -> n c t h w', t=self.frames_per_clip)
        clips_n = einops.rearrange(frames_n, 'b (n t) c h w -> b n c t h w', t=self.frames_per_clip)

        return anc, pos, neg, clips_a, clips_p, clips_n, n_clips_a, n_clips_p, n_clips_n

    @staticmethod
    def collate(batch):
        anc, pos, neg, clips_a, clips_p, clips_n, n_clips_a, n_clips_p, n_clips_n = tuple(zip(*batch))
        # neg = np.stack(neg)
        neg = np.concatenate(neg)
        clips_a = torch.nn.utils.rnn.pad_sequence(clips_a, batch_first=True)
        clips_p = torch.nn.utils.rnn.pad_sequence(clips_p, batch_first=True)
        # clips_n = torch.nn.utils.rnn.pad_sequence(clips_n)
        clips_n = torch.cat(clips_n)
        # n_clips_n = torch.stack(default_collate(n_clips_n))
        n_clips_n = torch.cat(default_collate(n_clips_n))

        return (default_collate(anc),
                default_collate(pos),
                neg,
                clips_a,
                clips_p,
                clips_n,
                default_collate(n_clips_a),
                default_collate(n_clips_p),
                n_clips_n)


class VideoDataset_VCDBDistraction(VideoDataset):
    def __init__(self, vcdb, transform_1, transform_2=None, *args, **kwargs):
        kwargs['transform'] = None
        super(VideoDataset_VCDBDistraction, self).__init__(*args, **kwargs)

        self.distractions = [v for v in vcdb.distraction if v in self.index.keys()]

        self.transform_1 = transform_1
        self.transform_2 = transform_2 if transform_2 else transform_1

    def __len__(self):
        return len(self.distractions)

    def __getitem__(self, idx):
        key = self.distractions[idx]

        video = self.index[key]

        frames, n_clips = self.load_clips(video, self.frames_per_clip, self.n_clips)

        frames_q = self.apply_transform(frames, self.transform_1)
        frames_k = self.apply_transform(frames, self.transform_2)

        clips_q = einops.rearrange(frames_q, '(n t) c h w -> n c t h w', t=self.frames_per_clip)
        clips_k = einops.rearrange(frames_k, '(n t) c h w -> n c t h w', t=self.frames_per_clip)

        return key, clips_q, clips_k, n_clips

    @staticmethod
    def collate(batch):
        key, clips_q, clips_k, n_clips_a = tuple(zip(*batch))
        clips_q = torch.nn.utils.rnn.pad_sequence(clips_q, batch_first=True)
        clips_k = torch.nn.utils.rnn.pad_sequence(clips_k, batch_first=True)
        return (default_collate(key),
                clips_q,
                clips_k,
                default_collate(n_clips_a)
                )


class VideoDataset_DEPRECATED(Dataset):
    def __init__(self, index: Union[dict, str, Path], frames_per_clip, sampling_rate, sampling_unit='frames',
                 nb_clips=None, transform=None):
        self.index = load_index(index)
        self.keys = sorted(self.index.keys())
        self.frames_per_clip = frames_per_clip
        self.sampling_rate = sampling_rate
        self.sampling_unit = sampling_unit.lower()
        assert self.sampling_unit in ['frames', 'fps']
        self.transform = transform
        self.FRAME_NAME_FORMAT = '{:06d}.jpg'
        self.nb_clips = nb_clips  # number of clips per video

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        info = self.index[key]

        if self.sampling_unit == 'fps':
            fps = Fraction(info.get('average_rate'))
            sampling_rate = round(Fraction(fps, self.sampling_rate))
        else:
            sampling_rate = self.sampling_rate

        if self.nb_clips:
            pass

        frame_dir = info.get('frames')
        if frame_dir:
            # Get frames from directory
            nb_frames = info['nb_frames']
            nb_clips = max(round(nb_frames / (self.frames_per_clip * sampling_rate)), 1)

            indices = np.arange(0, nb_clips * self.frames_per_clip) * sampling_rate
            indices = np.clip(indices, 0, nb_frames - 1)
            paths = list(map(lambda x: os.path.join(frame_dir, self.FRAME_NAME_FORMAT.format(x + 1)), indices))

            frames = self.load_images(paths)

        else:
            # Decode video
            frames = NotImplementedError

        if self.transform:
            frames = self.apply_transform(frames)

        frames = einops.rearrange(frames, '(n t) c h w -> n c t h w', n=nb_clips, t=self.frames_per_clip)

        return key, frames, nb_clips

    def load_images(self, paths):
        return [pil_loader(path) for path in paths]

    def apply_transform(self, frames):
        aug = self.transform(image=np.asarray(frames[0]))['replay']
        frames = [self.transform.replay(aug, image=np.asarray(f))['image'] for f in frames]

        frames = torch.stack(frames, dim=0)

        return frames

    @staticmethod
    def collate(batch):
        keys, frames, nb_clips = tuple(zip(*batch))
        frames = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
        return default_collate(keys), frames, default_collate(nb_clips)


FIVR_PKL = '/workspace/datasets/fivr.tca.pickle'

VCDB_PKL = '/workspace/datasets/vcdb.pkl'


class VCDB(object):
    def __init__(self, pickle=VCDB_PKL):
        dataset = load_pickle(pickle)
        self.pair = [tuple(p['videos']) for p in dataset['pair']]
        self.distraction = sorted(dataset['distraction'])

        self.keys = set(itertools.chain(*self.pair))

    def sampling_negative(self, count=1):
        return np.random.choice(self.distraction, count, replace=False)


class FIVR(object):
    def __init__(self, version, annotation=FIVR_PKL):
        assert version in ['5k', '200k']
        self.version = version

        _annotation = load_index(annotation)

        self._queries = sorted(_annotation[version]['queries'])
        self._database = sorted(_annotation[version]['database'])
        self._annotation = self._load_annotation(_annotation['annotation'])

    @property
    def annotation(self):
        return self._annotation

    @property
    def queries(self):
        return self._queries

    @property
    def database(self):
        return self._database

    def _load_annotation(self, annotation):
        if self.version == '200k':
            return annotation
        else:
            queries, database = set(self.queries), set(self.database)

            _annotation = {q: {k: database.intersection(gt) for k, gt in annotation[q].items()} for q in queries}
            return _annotation

    def get_groundtruth(self, query, task=None, labels=None, flatten=True):
        assert not (task and labels)  # mutex

        gt = self.annotation.get(query, [])
        if task or labels:
            if task.lower() == 'dsvr':
                labels = ['ND', 'DS']
            elif task.lower() == 'csvr':
                labels = ['ND', 'DS', 'CS']
            elif task.lower() == 'isvr':
                labels = ['ND', 'DS', 'CS', 'IS']
            else:
                labels = []
            gt = {label: gt[label] for label in labels if len(gt.get(label, []))}

        if flatten:
            gt = set(itertools.chain(*gt.values()))
        return gt

    def evaluate(self, features, metric='chamfer'):
        keys = set(features.keys())
        queries = sorted(list(keys.intersection(self.queries)))
        database = sorted(list(keys.intersection(self.database)))

        q_features = [features[q] for q in queries]
        db_features = [features[db] for db in database]
        if metric == 'chamfer':  # cilps to clips
            dist = chamfer_distance(q_features, db_features)
        elif metric == 'cosine':  # video to video
            q_features = np.concatenate(q_features, axis=0)
            db_features = np.concatenate(db_features, axis=0)
            dist = cosine_distance(q_features, db_features, backend='einops')
        else:
            raise NotImplementedError

        dsvr, csvr, isvr = .0, .0, .0
        indices = np.argsort(dist, axis=1)
        for n, query in enumerate(queries):
            dsvr_gt = {database.index(g) for g in self.get_groundtruth(query, 'dsvr') if g in database}
            csvr_gt = {database.index(g) for g in self.get_groundtruth(query, 'csvr') if g in database}
            isvr_gt = {database.index(g) for g in self.get_groundtruth(query, 'isvr') if g in database}
            dsvr += self.average_precision(indices[n, :], dsvr_gt)
            csvr += self.average_precision(indices[n, :], csvr_gt)
            isvr += self.average_precision(indices[n, :], isvr_gt)

        dsvr /= len(queries)
        csvr /= len(queries)
        isvr /= len(queries)

        return (dsvr, csvr, isvr)

    def average_precision(self, results, gts):
        c, ap = 0, 0.
        for n, r in enumerate(results, start=1):
            if r in gts:
                c += 1
                ap += c / n

            # find all groundtruth
            if c == len(gts):
                break
        ap /= len(gts)
        return ap


if __name__ == '__main__':

    fivr = FIVR('200k')
    queries = fivr.queries
    print(queries)

    gts = []
    pair = 0
    for q in queries:
        g = fivr.get_groundtruth(q, task='dsvr')
        print(q, g)
        pair += len(g)
        gts.append(g)
    print(len(queries))
    import itertools

    gts = set(itertools.chain(*gts)).union(set(queries))
    print(len(gts))
    print(pair, pair)
    index = load_json('fivr5k_info.json')
    d = 0.
    c = 0
    for p in gts:
        if index.get(p):
            d += index[p]['duration']
            c += 1
    print(d / c, d, c)
    print(d * len(gts) / c / len(gts))
    exit()
    transform = A.ReplayCompose([
        A.SmallestMaxSize(160),
        A.CenterCrop(160, 160),
        A.Normalize(),
        ToTensorV2()
    ])
    dataset = VideoDataset('fivr5k_info.json',
                           frames_per_clip=1,
                           sampling_rate=1,
                           sampling_unit='fps',
                           transform=transform
                           )

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate, num_workers=4)

    print(dataset.__len__())
    print(dataset.keys)

    for k, f, l in loader:
        print(k, f.shape, l)
