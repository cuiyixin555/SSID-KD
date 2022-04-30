import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, logstd):
        dim = mu.shape[1]
        mu = mu.permute(0, 2, 3, 1).contiguous()
        logstd = logstd.permute(0, 2, 3, 1).contiguous()
        mu = mu.view(-1, dim)
        logstd = logstd.view(-1, dim)

        std = torch.exp(logstd)
        kl = torch.sum(-logstd + 0.5 * (std ** 2 + mu ** 2), dim=-1) - (0.5 * dim)

        return kl