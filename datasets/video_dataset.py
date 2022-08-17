from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset, DataLoader
import torch

from torchvision.datasets.folder import pil_loader
from pytorchvideo.data.encoded_video import EncodedVideo

from typing import Union
from pathlib import Path
import numpy as np
import einops
import os

from PIL import Image, ImageFile
import albumentations as A

from .utils import load_index, get_clip_indices_from_video, get_clip_indices_from_frames

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoDataset(Dataset):
    def __init__(self,
                 index: Union[Path, str, dict],
                 frames_per_clip: int,
                 sampling_rate: Union[int, float],
                 sampling_unit: str = 'frames',
                 n_clips: int = None,
                 transform: A.ReplayCompose = None):
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

        n_clips, frames = self.read_frames(info, self.frames_per_clip, self.n_clips)

        if self.transform:
            frames = self.apply_transform(frames, self.transform)
        clips = einops.rearrange(frames, '(n t) c h w -> n c t h w', t=self.frames_per_clip)

        return key, clips, n_clips

    def read_frames(self, info, frames_per_clip=1, n_clip=None):
        if self.sampling_unit == 'frames':
            sampling_rate = self.sampling_rate
        else:
            sampling_rate = round(info.get('average_rate') / self.sampling_rate)

        frame_dir = info.get('frames')
        if frame_dir:
            n_clips, indices = get_clip_indices_from_frames(info,
                                                            sampling_rate=sampling_rate,
                                                            frames_per_clip=frames_per_clip,
                                                            n_clip=n_clip)
            indices = np.clip(indices, 0, info['nb_frames'] - 1)

            frames = [pil_loader(os.path.join(frame_dir, self.FRAME_NAME_FORMAT.format(idx + 1))) for idx in indices]

        else:
            n_clips, start_sec, end_sec, indices = get_clip_indices_from_video(info,
                                                                               sampling_rate=sampling_rate,
                                                                               frames_per_clip=frames_per_clip,
                                                                               n_clip=n_clip)
            video = EncodedVideo.from_path(info['video'], decode_audio=False, decoder='pyav')
            frames = video.get_clip(start_sec, end_sec + 1e-6)['video']

            indices = torch.tensor(np.clip(indices, 0, frames.shape[1] - 1))
            frames = torch.index_select(frames, dim=1, index=indices)
            frames = einops.rearrange(frames, 'c t h w -> t h w c')

        return n_clips, frames

    def apply_transform(self, frames, transform):
        aug = transform(image=np.asarray(frames[0]))['replay']
        frames = [transform.replay(aug, image=np.asarray(f))['image'] for f in frames]

        frames = torch.stack(frames, dim=0)
        return frames

    @staticmethod
    def collate(batch):
        keys, frames, nb_clips = tuple(zip(*batch))
        frames = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
        return default_collate(keys), frames, default_collate(nb_clips)
