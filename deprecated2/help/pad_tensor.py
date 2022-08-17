import torch
import torch.nn.functional as F

# n c t h w
a = torch.rand((4, 3, 2, 2, 2))
print(a)

b = F.pad(a, (1, 2), 'constant', value=0)
print(b)
print(b.shape)

a = torch.rand((5, 2))
b = torch.rand((5, 2))
c = torch.nn.utils.rnn.pad_sequence([a, b],batch_first=True)
print(c.shape)
print(c[0].shape)
print(c[1].shape)
print(c[0])