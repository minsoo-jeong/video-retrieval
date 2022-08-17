import torch
import torch.nn.functional as F

import einops

a = torch.rand(2, 100)

b = torch.rand(3, 100)

naive = torch.tensor(
    [[a[i].dot(b[j]) / (a[i].norm(p=2, dim=0) * b[j].norm(p=2, dim=0)) for j in range(b.shape[0])] for i in
     range(a.shape[0])])
print(naive)
an = F.normalize(a, p=2, dim=1)
bn = F.normalize(b, p=2, dim=1)

norm_tensor_dot = torch.einsum('ik,jk->ij', [an, bn])
print(norm_tensor_dot)
norm_tensor_dot2 = an.matmul(bn.transpose(1, 0))

print(norm_tensor_dot2)

print(torch.max(naive,dim=1)[0].sum()/naive.shape[0])
print(naive.max(dim=1))
