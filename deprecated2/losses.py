import torch


class CircleLoss(torch.nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        alpha = torch.clamp_min(logits + self.m, min=0).detach()
        alpha[labels] = torch.clamp_min(-logits[labels] + 1 + self.m, min=0).detach()
        delta = torch.ones_like(logits, device=logits.device, dtype=logits.dtype) * self.m
        delta[labels] = 1 - self.m
        return self.loss(alpha * (logits - delta) * self.gamma, labels)
