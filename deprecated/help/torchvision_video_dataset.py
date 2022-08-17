from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset

from torchvision.datasets import Kinetics400, Kinetics

from pathlib import Path



# videos = [p.as_posix() for p in Path('/mldisk/nfs_shared_/MLVD/VCDB/videos/core_dataset/').rglob('*.mp4')]
# vc = VideoClips(videos)

# print('valid')
# Kinetics(root='/mlsun/kinetics400/valid', split='val', frames_per_clip=1, num_download_workers=8, download=False)
print('train')
Kinetics(root='/mlsun/kinetics400/train', split='train', frames_per_clip=1,
         num_workers=8,num_download_workers=8, download=False)
