import datetime

import cv2
import numpy as np
import multiprocessing as mp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import transforms as trn
from torchvision import models

from tqdm import tqdm
from collections import defaultdict, OrderedDict

import faiss

from pymediainfo import MediaInfo
import subprocess
from PIL import Image
from torchvision.transforms import functional as TF

from typing import Union


def safe_divide(a, b, eps=1e-6):
    return (a) / (b + eps)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RatioMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = AverageMeter()
        self.gt = AverageMeter()
        self.val = 0.
        self.acc = 0.

        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, correct, gt, n=1):
        self.correct.update(correct, n=n)
        self.gt.update(gt, n=n)

        self.val = safe_divide(self.correct.val, self.gt.val)
        self.acc = safe_divide(self.correct.sum, self.gt.sum)

        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class FscoreMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.precision = RatioMeter()
        self.recall = RatioMeter()
        self.val = 0.
        self.acc = 0.

        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, d_c, d_a, g_c, g_a, n=1):
        self.precision.update(d_c, d_a, n=n)
        self.recall.update(g_c, g_a, n=n)
        self.val = safe_divide(2 * self.precision.val * self.recall.val, self.precision.val + self.recall.val)
        self.acc = safe_divide(2 * self.precision.acc * self.recall.acc, self.precision.acc + self.recall.acc)

        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class Interval(object):
    # half-closed form [a, b)
    def __init__(self, t1, t2):
        self.start, self.end = (t1, t2) if t1 < t2 else (t2, t1)

    def __repr__(self):
        return '{} - {}'.format(self.start, self.end)

    @property
    def length(self):
        return self.end - self.start

    def __add__(self, v: Union[int, float]):
        self.start += v
        self.end += v
        return self

    def __sub__(self, v: Union[int, float]):
        self.start -= v
        self.end -= v
        return self

    def __mul__(self, v: Union[int, float]):
        self.start *= v
        self.end *= v
        return self

    def is_overlap(self, o):
        assert isinstance(o, Interval)
        return not ((self.end <= o.start) or (o.end <= self.start))

    def is_in(self, o):
        assert isinstance(o, Interval)
        return o.start <= self.start and self.end <= o.end

    # self.start <= o.start <= o.end <= self.end
    def is_wrap(self, o):
        assert isinstance(o, Interval)
        return self.start <= o.start and o.end <= self.end

    def intersection(self, o):
        assert isinstance(o, Interval)
        return Interval(max(self.start, o.start), min(self.end, o.end)) if self.is_overlap(o) else None

    # if not overlap -> self
    def union(self, o):
        assert isinstance(o, Interval)
        return Interval(min(self.start, o.start), max(self.end, o.end)) if self.is_overlap(o) else None

    def IOU(self, o):
        try:
            intersect = self.intersection(o)
            union = self.union(o)
            iou = intersect.length / union.length
        except:
            iou = 0
        return iou


class TemporalNetwork(object):
    def __init__(self, D, video_idx, frame_idx, time_window=10, min_match=5, min_score=-1, numpy=True):
        self.time_window, self.min_match, self.min_score = time_window, min_match, min_score
        self.numpy = numpy

        # [# of query index, topk]
        self.video_index = video_idx
        self.frame_index = frame_idx
        self.dist = D

        self.query_length = D.shape[0]
        self.topk = D.shape[1]

        # dist, count, query start, reference start
        self.paths = np.empty((*D.shape, 4), dtype=object)

    def find_previous_linkable_nodes(self, t, r):
        video_idx, frame_idx = self.video_index[t, r], self.frame_index[t, r]
        min_prev_time = max(0, t - self.time_window)

        # find previous nodes that have (same video index) and (frame timestamp - wnd <= previous frame timestamp < frame timestamp)
        time, rank = np.where((self.dist[min_prev_time:t, ] >= self.min_score) &
                              (self.video_index[min_prev_time:t, ] == video_idx) &
                              (self.frame_index[min_prev_time:t, ] >= frame_idx - self.time_window) &
                              (self.frame_index[min_prev_time:t, ] < frame_idx)
                              )

        return np.stack((time + min_prev_time, rank), axis=-1)

    def find_previous_linkable_nodes_naive(self, t, r):
        video_idx, frame_idx = self.video_index[t, r], self.frame_index[t, r]
        min_prev_time = max(0, t - self.time_window)
        nodes = []
        for prev_time in range(min_prev_time, t):
            for k in range(self.topk):
                if self.dist[prev_time, k] >= self.min_score and \
                        self.video_index[prev_time, k] == video_idx and \
                        (frame_idx - self.time_window) <= self.frame_index[prev_time, k] < frame_idx:
                    nodes.append([prev_time, k])
        return np.array(nodes)

    def fit(self):
        # find linkable nodes
        for time in range(self.query_length):
            for rank in range(self.topk):
                # if self.dist[time,rank]>self.min_score:
                #     if self.numpy:
                #         prev_linkable_nodes = self.find_previous_linkable_nodes(time, rank)
                #     else:
                #         prev_linkable_nodes = self.find_previous_linkable_nodes_naive(time, rank)
                # else:
                #     prev_linkable_nodes=[]
                if self.numpy:
                    prev_linkable_nodes = self.find_previous_linkable_nodes(time, rank)
                else:
                    prev_linkable_nodes = self.find_previous_linkable_nodes_naive(time, rank)
                if not len(prev_linkable_nodes):
                    self.paths[time, rank] = [self.dist[time, rank],
                                              1,
                                              time,
                                              self.frame_index[time, rank]]
                else:
                    # priority : count, path length, path score
                    prev_time, prev_rank = max(prev_linkable_nodes, key=lambda x: (self.paths[x[0], x[1], 1],
                                                                                   self.frame_index[time, rank] -
                                                                                   self.paths[x[0], x[1], 3],
                                                                                   self.paths[x[0], x[1], 0]
                                                                                   ))
                    prev_path = self.paths[prev_time, prev_rank]
                    self.paths[time, rank] = [prev_path[0] + self.dist[time, rank],
                                              prev_path[1] + 1,
                                              prev_path[2],
                                              prev_path[3]]

        # connect and filtering paths
        candidate = defaultdict(list)
        for time in reversed(range(self.query_length)):
            for rank in range(self.topk):
                score, count, q_start, r_start = self.paths[time, rank]
                if count >= self.min_match:
                    video_idx, frame_idx = self.video_index[time, rank], self.frame_index[time, rank]
                    q = Interval(q_start, time + 1)
                    r = Interval(r_start, frame_idx + 1)
                    path = (video_idx, q, r, score, count)
                    flag = True
                    for n, c in enumerate(candidate[video_idx]):
                        if path[1].is_wrap(c[1]) and path[2].is_wrap(c[2]):
                            candidate[video_idx][n] = path
                            flag = False
                            break
                        elif path[1].is_in(c[1]) and path[2].is_in(c[2]):
                            flag = False
                            break
                    if flag:
                        candidate[video_idx].append(path)

        # remove overlap path
        for video, path in candidate.items():
            candidate[video] = self.nms_path(path)

        # candidate = [[c[0], c[1], c[2], c[3], c[4]] for cc in candidate.values() for c in cc]
        candidate = [c for cc in candidate.values() for c in cc]

        return candidate

    def nms_path(self, path):
        l = len(path)
        path = np.array(sorted(path, key=lambda x: (x[4], x[3], x[2].length, x[1].length), reverse=True))

        keep = np.array([True] * l)
        overlap = np.vectorize(lambda x, a: x.is_overlap(a))
        for i in range(l - 1):
            if keep[i]:
                keep[i + 1:] = keep[i + 1:] & \
                               (~(overlap(path[i + 1:, 1], path[i, 1]) & overlap(path[i + 1:, 2], path[i, 2])))
        path = path.tolist()
        path = [path[n] for n in range(l) if keep[n]]

        return path


def matching_with_annotation(detect, gt, videos, intv_per_feature):
    def _match(al, bl):
        c = 0
        for a in al:
            for b in bl:
                if a[0] == b[0] and a[1].is_overlap(b[1]) and a[2].is_overlap(b[2]):
                    c += 1
                    break
        return c

    detect = [[videos[r[0]], r[1] * intv_per_feature, r[2] * intv_per_feature, *r[3:]] for r in detect]

    d = _match(detect, gt)
    g = _match(gt, detect)

    return d, g


def read_video_files(files):
    def _toabs(p, root='/'):
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(root, p))

    videos = OrderedDict()
    for file in files:
        root_dir = os.path.dirname(os.path.abspath(file))
        for n, line in enumerate(open(file, 'r').read().splitlines()):
            v_path, *f_root = list(map(lambda x: _toabs(x.strip(), root_dir), line.split(',')))

            # ignore same basename(accept first)
            key = os.path.basename(v_path)
            if key not in videos.keys():
                videos[key] = {'video': v_path, 'frame_root': f_root[0] if len(f_root) else None}
            else:
                print(f'[KeyDuplicate] Ignore {file} line {n}: {v_path}')

    return videos


def read_annotation_files(files):
    annotations = defaultdict(list)
    for file in files:
        for line in open(file, 'r').readlines():
            a, b, *timestamps = list(map(lambda x: x.strip(), line.split(',')))
            timestamps = list(map(int, timestamps))
            annotations[a].append([b, Interval(*timestamps[:2]), Interval(*timestamps[2:])])
            if a != b:
                annotations[b].append([a, Interval(*timestamps[2:]), Interval(*timestamps[:2])])

    return annotations


def find_feature_paths(roots, keys):
    all_paths = dict()
    for root in roots:
        for r, dirs, files in os.walk(root):
            # p = {os.path.splitext(f)[0]: os.path.join(r, f) for f in files if os.path.splitext(f)[0] in keys}
            all_paths.update({os.path.splitext(f)[0]: os.path.join(r, f) for f in files})

    paths = {k: all_paths[k] for k in keys if k in all_paths.keys()}

    return paths


def load_features(paths):
    bar = tqdm(paths.items(), desc='Load features', ncols=100, mininterval=1)
    features = {k: np.load(p) for k, p in bar}

    return features


def build_index(features, gpu=True):
    _features = np.concatenate([f for f in features.values()])
    _index = faiss.IndexFlatIP(_features.shape[1])
    _index.add(_features)

    if gpu:
        res = faiss.StandardGpuResources()
        _index = faiss.index_cpu_to_gpu(res, 1, _index)
        # _index = faiss.index_cpu_to_all_gpus(_index)

    _ids = ([(n, i) for n, f in enumerate(features.values()) for i in range(f.shape[0])])
    _idx_to_id = np.vectorize(lambda x: _ids[x])
    return _index, _idx_to_id


class L2N(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(eps={self.eps})'


class MobileNet_AVG(nn.Module):
    def __init__(self):
        super(MobileNet_AVG, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class Segment_AvgPool(nn.Module):
    def __init__(self):
        super(Segment_AvgPool, self).__init__()
        self.norm = L2N()

    def forward(self, x):
        x = torch.mean(x, 1)
        x = self.norm(x)
        return x


class ListDataset(Dataset):
    def __init__(self, l):
        self.l = l
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        im = self.l[idx]
        frame = self.transform(im)
        return frame

    def __len__(self):
        return len(self.l)


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class ListDataset_rotate(Dataset):
    def __init__(self, l):
        self.l = l
        self.transforms = [
            trn.Compose([
                trn.Resize((224, 224)),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])]
        self.transforms += [
            trn.Compose([
                trn.Resize((224, 224)),
                RotationTransform(i),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]) for i in range(90, 360, 90)]

    def __getitem__(self, idx):
        im = self.l[idx]
        images = [tr(im) for tr in self.transforms]
        return images

    def __len__(self):
        return len(self.l)


def decode_video(video, sampling_rate=1, target_size=None):
    media_info = MediaInfo.parse(video)
    metadata = {'file_path': video}
    for track in media_info.tracks:
        if track.track_type == 'General':
            metadata['file_name'] = track.file_name + '.' + track.file_extension
            metadata['file_extension'] = track.file_extension
            metadata['format'] = track.format
        elif track.track_type == 'Video':
            metadata['width'] = int(track.width)
            metadata['height'] = int(track.height)
            metadata['rotation'] = float(track.rotation or 0.)
            metadata['codec'] = track.codec

    frames = []
    w, h = (metadata['width'], metadata['height']) if metadata['rotation'] not in [90, 270] else (
        metadata['height'], metadata['width'])
    command = ['ffmpeg',
               '-hide_banner', '-loglevel', 'panic',
               '-vsync', '2',
               '-i', video,
               '-pix_fmt', 'bgr24',  # color space
               '-vf', f'fps={sampling_rate}',  # '-r', str(decode_rate),
               '-q:v', '0',
               '-vcodec', 'rawvideo',  # origin video
               '-f', 'image2pipe',  # output format : image to pipe
               'pipe:1']
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=w * h * 3)
    while True:
        raw_image = pipe.stdout.read(w * h * 3)
        pipe.stdout.flush()
        try:
            image = Image.frombuffer('RGB', (w, h), raw_image, "raw", 'BGR', 0, 1)
        except ValueError as e:
            break

        if target_size is not None:
            image = TF.resize(image, target_size)
        frames.append(image)
    return frames


@torch.no_grad()
def extract_feature(model, aggr_model, video, batch=4, seg=5, progress=False):
    model.eval()
    frames = decode_video(video)

    loader = DataLoader(ListDataset(frames), batch_size=batch, shuffle=False, num_workers=4)
    frame_features = []
    if progress:
        bar = tqdm(loader, ncols=200)
    for i, image in enumerate(loader):
        out = model(image.cuda())
        frame_features.append(out)
        if progress:
            bar.update()
    frame_features = torch.cat(frame_features)
    if progress:
        bar.close()

    seg_features = []
    for ff in torch.split(frame_features, seg):
        out = aggr_model(ff.unsqueeze(0)).cpu()
        seg_features.append(out)
    seg_features = torch.cat(seg_features).numpy()

    return seg_features


@torch.no_grad()
def extract_feature_rotate(model, aggr_model, video, batch=4, seg=5, progress=False):
    frames = decode_video(video)
    loader = DataLoader(ListDataset_rotate(frames), batch_size=batch, shuffle=False, num_workers=4)
    if progress:
        bar = tqdm(loader, ncols=200)

    frame_features = [[], [], [], []]
    for i, images in enumerate(loader):
        _images = torch.cat(images)

        out = model(_images.cuda())
        _out = torch.chunk(out, 4)
        for n, f in enumerate(_out):
            frame_features[n].append(f)
        if progress:
            bar.update()
    frame_features = [torch.cat(f) for f in frame_features]
    if progress:
        bar.close()

    seg_features = [[], [], [], []]
    for n, feature in enumerate(frame_features):
        for ff in torch.split(feature, seg):
            out = aggr_model(ff.unsqueeze(0)).cpu()
            seg_features[n].append(out)
        seg_features[n] = torch.cat(seg_features[n]).numpy()

    return seg_features


@torch.no_grad()
def extract_feature_rotate2(model, aggr_model, video, batch=4, seg=5, progress=False):
    frames = decode_video(video)
    loader = DataLoader(ListDataset_rotate(frames), batch_size=batch, shuffle=False, num_workers=4)
    if progress:
        bar = tqdm(loader, ncols=200)

    frame_features = [[], [], [], []]
    for i, images in enumerate(loader):
        for n in range(4):
            out = model(images[n].cuda())
            frame_features[n].append(out)
        if progress:
            bar.update()

    frame_features = [torch.cat(f) for f in frame_features]
    if progress:
        bar.close()

    seg_features = [[], [], [], []]
    for n, feature in enumerate(frame_features):
        for ff in torch.split(feature, seg):
            out = aggr_model(ff.unsqueeze(0)).cpu()
            seg_features[n].append(out)
        seg_features[n] = torch.cat(seg_features[n]).numpy()

    return seg_features


def test():
    model = MobileNet_AVG().cuda()
    model.load_state_dict(torch.load('mobilenet_avg_ep16_ckpt.pth')['model_state_dict'])
    model = model.eval()
    aggr_model = Segment_AvgPool().cuda()
    aggr_model.eval()

    v = 'sample.mp4'
    all_feature = extract_feature_rotate(model, aggr_model, v, batch=4, seg=5)
    all_feature2 = extract_feature_rotate2(model, aggr_model, v, batch=4, seg=5)
    r0 = extract_feature(model, aggr_model, v, batch=4, seg=5)

    v = 'sample_90.mp4'
    r90 = extract_feature(model, aggr_model, v, batch=4, seg=5)

    v = 'sample_180.mp4'
    r180 = extract_feature(model, aggr_model, v, batch=4, seg=5)

    v = 'sample_270.mp4'
    r270 = extract_feature(model, aggr_model, v, batch=4, seg=5)

    print(f'[R 0] {np.mean(np.abs(all_feature[0] - r0)):e} {np.max(np.abs(all_feature[0] - r0)):e}')
    print(f'[R 0] {np.mean(np.abs(all_feature2[0] - r0)):e} {np.max(np.abs(all_feature2[0] - r0)):e}')

    print(f'[R 90] {np.mean(np.abs(all_feature[1] - r90)):e} {np.max(np.abs(all_feature[1] - r90)):e}')
    print(f'[R 180] {np.mean(np.abs(all_feature[2] - r180)):e} {np.max(np.abs(all_feature[2] - r180)):e}')
    print(f'[R 270] {np.mean(np.abs(all_feature[3] - r270)):e} {np.max(np.abs(all_feature[3] - r270)):e}')


def extract_vcdb_segment_features(target_root):
    vcdb_videos = read_video_files(['data/vcdb.txt'])
    model = MobileNet_AVG().cuda()
    model.load_state_dict(torch.load('mobilenet_avg_ep16_ckpt.pth')['model_state_dict'])
    model = model.eval()

    aggr_model = Segment_AvgPool().cuda()
    aggr_model.eval()
    for k, v in vcdb_videos.items():
        video_path = v['video']
        seg_feature = extract_feature(model, aggr_model, video_path, batch=4)
        target_path = os.path.join(target_root, f'{k}.npy')
        np.save(target_path, seg_feature)


if __name__ == '__main__':
    # extract_vcdb_segment_features('/workspace/features')
    video_list_file = 'data/vcdb.txt'
    annotation_file = 'data/vcdb-annotation.txt'
    feature_root_dir = '/workspace/features'
    TopK, TN_window, TN_match = 50, 5, 3

    vcdb_videos = read_video_files([video_list_file])
    annotations = read_annotation_files([annotation_file])
    db_videos = list(vcdb_videos.keys())
    db_paths = find_feature_paths([feature_root_dir], vcdb_videos.keys())

    db_features = load_features(db_paths)
    print('Build index', end='')
    db_index, _idx_to_id = build_index(db_features, gpu=True)
    print(f'\rBuild index: ({db_index.ntotal}, {db_index.d})')

    model = MobileNet_AVG().cuda()
    model.load_state_dict(torch.load('mobilenet_avg_ep16_ckpt.pth')['model_state_dict'])
    model = model.eval()
    aggr_model = Segment_AvgPool().cuda()
    aggr_model.eval()

    fscore = FscoreMeter()
    fscore2 = FscoreMeter()
    dec = AverageMeter()  # decoding
    fe1 = AverageMeter()  # feature extract
    fe2 = AverageMeter()
    ns1 = AverageMeter()  # nearest search
    ns2 = AverageMeter()

    q_keys = vcdb_videos.keys()
    progress = tqdm(q_keys, desc='Detect Copy Segment', ncols=100)
    for n, q in enumerate(progress, start=1):
        annotation = annotations[q]
        q_video_path = vcdb_videos[q]['video']

        #####################
        #   Single Query
        #####################
        now = datetime.datetime.now()
        q_feat = extract_feature(model, aggr_model, q_video_path, batch=32, seg=5)
        t1 = datetime.datetime.now() - now
        fe1.update(t1.total_seconds())

        now = datetime.datetime.now()
        dist, idx = db_index.search(q_feat, TopK)
        t2 = datetime.datetime.now() - now
        ns1.update(t2.total_seconds())

        video_id, frame_id = _idx_to_id(idx)
        result = TemporalNetwork(dist, video_id, frame_id, time_window=TN_window, min_match=TN_match).fit()
        d, g = matching_with_annotation(result, annotation, db_videos, intv_per_feature=5)

        fscore.update(d, len(result), g, len(annotation))

        #####################
        #   Rotated Query
        #####################

        # Extract
        now = datetime.datetime.now()
        q_feat_rotated = extract_feature_rotate(model, aggr_model, q_video_path, batch=32, seg=5)
        t3 = datetime.datetime.now() - now
        fe2.update(t3.total_seconds())

        # search
        now = datetime.datetime.now()
        q_feat_rotated = np.concatenate(q_feat_rotated)  # concat rotated feature, [N*4, dim]
        r_dist, r_idx = db_index.search(q_feat_rotated, TopK)  # nearest search [N*4, TopK]
        r_dist = np.concatenate(np.vsplit(r_dist, 4), axis=-1)  # reshape [N,TopK * 4]
        r_idx = np.concatenate(np.vsplit(r_idx, 4), axis=-1)

        # remove duplicated and get topk index, dist
        dist, idx = [], []
        sorted_index = np.argsort(r_dist)[:, ::-1]  # sort by dist
        sorted_r_idx = np.take_along_axis(r_idx, sorted_index, axis=1)
        sorted_r_dist = np.take_along_axis(r_dist, sorted_index, axis=1)

        # np.unique remove duplicated value (remain first)
        for nr, row in enumerate(sorted_r_idx):
            unique_r_idx, unique_index = np.unique(row, return_index=True)  # return sorted by idx value
            unique_r_dist = sorted_r_dist[nr][unique_index]

            resorted_index = np.argsort(unique_r_dist)[::-1][:TopK]  # re-sort by dist and filter out with TopK

            idx.append(unique_r_idx[resorted_index])
            dist.append(unique_r_dist[resorted_index])
        dist = np.array(dist)
        idx = np.array(idx)
        t4 = datetime.datetime.now() - now
        ns2.update(t4.total_seconds())

        video_id, frame_id = _idx_to_id(idx)
        result = TemporalNetwork(dist, video_id, frame_id, time_window=TN_window, min_match=TN_match).fit()
        d, g = matching_with_annotation(result, annotation, db_videos, intv_per_feature=5)

        fscore2.update(d, len(result), g, len(annotation))

        log = f'[{n}/{len(q_keys)} SQ] F-score: {fscore.val:.4f}({fscore.acc:.4f}) '
        log += f'Precision: {fscore.precision.val:.4f}({fscore.precision.acc:.4f}) '
        log += f'Recall: {fscore.recall.val:.4f}({fscore.recall.acc:.4f}) '
        log += f'Fe Time: {fe1.val:.4f}({fe1.avg:.4f}) NS Time: {ns1.val:.4f}({ns1.avg:.4f})\n'
        log += f'[{n}/{len(q_keys)} RQ] F-score: {fscore2.val:.4f}({fscore2.acc:.4f}) '
        log += f'Precision: {fscore2.precision.val:.4f}({fscore2.precision.acc:.4f}) '
        log += f'Recall: {fscore2.recall.val:.4f}({fscore2.recall.acc:.4f}) '
        log += f'Fe Time: {fe2.val:.4f}({fe2.avg:.4f}) NS Time: {ns2.val:.4f}({ns2.avg:.4f})'

        # Frame Decoding Time
        # now = datetime.datetime.now()
        # frames=decode_video(q_video_path)
        # t0 = datetime.datetime.now() - now
        # dec.update(t0.total_seconds())
        # log += f'Decoding Time: {dec.val:.4f}({dec.avg:.4f})'

        progress.write(log)

    progress.close()
    print(f'>> F-score: {fscore.acc:.4f}, Precision: {fscore.precision.acc:.4f}, Recall: {fscore.recall.acc:.4f}')
