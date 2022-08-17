import torch

from models.models import *
from models.Moco import *

if __name__ == '__main__':
    m = VideoMoCo(Resnet50(), SimpleMLP(2048, 1024))
    length = 16
    batch = 32
    a = torch.rand([batch, length, 3, 224, 224])
    b = torch.Tensor([length] * batch)
    with torch.no_grad():
        c = m(a, a, a, b, b, b)

