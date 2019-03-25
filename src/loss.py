from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Reference: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        distance = (output2 - output1).pow(2).sum(1)
        loss = 0.5 * ((1 - target).float() * distance.pow(2) +
                      (target).float() * F.relu(self.margin - distance).pow(2))
        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss function.
    Reference: https://arxiv.org/pdf/1503.03832.pdf
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        loss = F.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()
