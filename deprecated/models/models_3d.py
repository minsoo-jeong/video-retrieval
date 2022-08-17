import torch
import torch.nn as nn

import pytorchvideo
from torchsummary import summary


class TorchHubWrap(nn.Module):
    def __init__(self, model_name):
        super(TorchHubWrap, self).__init__()
        model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

        head = model.blocks[-1]
        if isinstance(head, pytorchvideo.models.head.ResNetBasicHead):
            projection_layer = head.pool
            model.blocks[-1] = projection_layer
        else:
            raise NotImplementedError('Unsupported model')

        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(-1).squeeze(-1).squeeze(-1)


if __name__ == '__main__':
    slow_r50 = ['slow_r50', 8, 8, 224, 224]  # train max-batch: 8 (13 GB)
    x3d_s = ['x3d_s', 13, 6, 160, 160]  # train max-batch: 16 (14 GB)
    x3d_m = ['x3d_m', 16, 5, 224, 224]  # train max-batch: 8 (17 GB)
    x3d_l = ['x3d_l', 16, 5, 312, 312]  # train max-batch: 2 (15 GB)

    cfg = x3d_l
    model = TorchHubWrap(cfg[0])
    frame_len, sampling_rate, frame_width, frame_height = cfg[1:]

    summary(model.cuda(), (3, frame_len, frame_width, frame_height))

    batch = 2

    input = torch.rand((batch, 3, frame_len, frame_width, frame_height)).cuda()
    for i in range(100):
        output = model(input)
        print(output.shape)
