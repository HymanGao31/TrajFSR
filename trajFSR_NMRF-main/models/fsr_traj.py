"""FSR separation + recalibration for agent embeddings [B, A, D].

Copied from TrajFSR (IMLE / MART plugin). NMRF applies it on History Encoder
outputs reshaped as [P, 1, 256].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.gumbel_sigmoid import GumbelSigmoid


class _FeatMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, feat):
        return self.net(feat)


class TrajFSR(nn.Module):
    def __init__(self, dim, hidden=128, tau=0.1):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.sep_net = _FeatMLP(dim, hidden)
        self.rec_net = _FeatMLP(dim, hidden)
        self.gumbel = GumbelSigmoid(tau=tau)

    def forward(self, feat, is_eval=False):
        """
        feat: [B, A, D]
        returns r_feat, nr_feat, rec_feat, mask, rec_units
        """
        b, a, d = feat.shape
        rob_map = self.sep_net(feat)

        mask = rob_map.reshape(b, 1, -1)
        mask = torch.sigmoid(mask)
        mask = self.gumbel(mask, is_eval=is_eval)
        mask = mask[:, 0].reshape(b, a, d)

        r_feat = feat * mask
        nr_feat = feat * (1.0 - mask)
        rec_units = self.rec_net(nr_feat) * (1.0 - mask)
        rec_feat = nr_feat + rec_units
        return r_feat, nr_feat, rec_feat, mask, rec_units
