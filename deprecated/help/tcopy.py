import torch
from torch.autograd import Variable


class T(torch.nn.Module):
    def __init__(self):
        super(T, self).__init__()

    def forward(self, x):
        g = x + 3

        o = x

        o[..., 2:4, 2:4] = g[..., 2:4, 2:4]
        f = torch.sum(o)
        return f


# def stitching(x, ori):


a = torch.zeros([5, 5], dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0)

m = T()
b = m(a)
print(b)

b.backward()
