import faiss
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from functools import partial


def cosine_similarity(a, b, backend):
    if backend == 'scipy':
        sims = 1 - cdist(a, b, metric='cosine')
    elif backend == 'faiss':
        na = a / np.linalg.norm(a, ord=2, axis=1, keepdims=True)
        nb = b / np.linalg.norm(b, ord=2, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(nb.shape[1])
        index.add(nb)
        dist, idx = index.search(na, k=nb.shape[0])
        sims = np.take_along_axis(dist, np.argsort(idx, axis=1), axis=1)

    elif backend == 'einops':
        na = F.normalize(torch.Tensor(a), p=2, dim=1)
        nb = F.normalize(torch.Tensor(b), p=2, dim=1)
        sims = torch.einsum('ik,jk->ij', [na, nb])
    else:
        raise NotImplementedError

    if isinstance(sims, torch.Tensor):
        sims = sims.numpy()

    return sims


def cosine_distance(a, b, backend):
    return 1 - cosine_similarity(a, b, backend)


def chamfer_distance(a, b, compare=partial(cosine_similarity, backend='einops')):
    def _chamfer(a, b, compare):
        dist = compare(a, b)
        # dist = dist.max(dim=1)[0].sum().cpu().item() / dist.shape[0]
        dist = np.sum(np.max(dist, axis=1)) / dist.shape[0]

        return 1 - dist

    dist = np.array([[_chamfer(q, d, compare) for d in b] for q in a])
    return dist


if __name__ == '__main__':
    a = np.random.rand(3, 5).astype(np.float32)
    print(a)

    print(cosine_similarity(a, a, backend='scipy'))
    print(cosine_similarity(a, a, backend='faiss'))
    print(cosine_similarity(a, a, backend='einops'))
