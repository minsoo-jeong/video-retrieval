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

from pytorchvideo import transforms as pv_transform
from pytorchvideo.data.encoded_video import EncodedVideo


class VideoFrameDataset(Dataset):
    def __init__(self, index, video_transform, frame_transform):
        super(VideoFrameDataset, self).__init__()
        self.index = load_pickle(index)


class VideoDataset(Dataset):
    def __init__(self, index, video_transform, frame_transform):
        super(VideoDataset, self).__init__()
        self.index = load_pickle(index)

    def __getitem__(self, idx):
        path = self.index[idx]
        video=EncodedVideo.from_path(path)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

    # dataset = VCDBTripletDataset('/workspace/vcdb_resnet50-avgpool.pickle', n_negative=16, length=20)
    dataset = VCDBDistractionDataset('/workspace/vcdb_frames_mlsun.pickle', length=32)
    loader = DataLoader(dataset, batch_size=16, num_workers=8)
    for k, f, len_f in loader:
        print(f.shape, len_f, k)
    exit()
