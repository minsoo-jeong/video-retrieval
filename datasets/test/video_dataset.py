from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import VideoDataset

if __name__ == '__main__':
    trn = A.ReplayCompose([
        A.Resize(160,160),
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ])
    dataset = VideoDataset('../../data/fivr5k.json',
                           frames_per_clip=1,
                           sampling_rate=1,
                           sampling_unit='fps',
                           transform=trn
                           )

    print(trn)
    print(type(trn))
    print(dataset)

    for n, (key, frames, n_clips) in enumerate(dataset):
        print(key, frames.shape, n_clips)

        if n == 10:
            break

    loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate)

    for n, (key, frames, n_clips) in enumerate(loader):
        print(key, frames.shape, n_clips)

        if n == 10:
            break
