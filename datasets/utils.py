from fractions import Fraction
from typing import Union
from pathlib import Path
import math
import numpy as np
import pickle as pk
import json
import os


def load_pickle(path):
    with open(path, 'rb') as f:
        content = pk.load(f)
    return content


def load_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content


def load_index(index: Union[str, dict, Path]):
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


def get_clip_indices_from_frames(info, sampling_rate=30, frames_per_clip=1, n_clip=None):
    len_clip = frames_per_clip * sampling_rate
    clips = max(round(info['nb_frames'] / len_clip), 1)  # total number of clips

    offset, n_frames = 0, clips * frames_per_clip

    if n_clip:
        n_frames = n_clip * frames_per_clip
        clips = min(clips, n_clip)
        if info['nb_frames'] > len_clip * clips:
            offset = np.random.choice(info['nb_frames'] - len_clip * clips)  # random offset

    indices = np.arange(0, n_frames) * sampling_rate + offset

    return clips, indices


def get_clip_indices_from_video(info, sampling_rate=30, frames_per_clip=1, n_clip=None):
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
