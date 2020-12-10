from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
import numpy as np


class PriorBox_kmeans(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox_kmeans, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        anchors_4 = np.array([
            [0.24861111, 0.05775862],
            [0.31388889, 0.075],
            [0.39166667, 0.08965517],
            [0.43333333, 0.12068966]
            ])
        anchors_6 = np.array([
            [0.23472222, 0.05431034],
            [0.29166667, 0.06896552],
            [0.35416667, 0.07931034],
            [0.32916667, 0.10948276],
            [0.425, 0.09396552],
            [0.46666667, 0.125]
            ])
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.feature_maps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                if len(self.aspect_ratios[k]) == 1:
                    for m in range(anchors_4.shape[0]):
                        mean += [cx, cy, anchors_4[m, 0], anchors_4[m, 1]]
                elif len(self.aspect_ratios[k]) == 2:
                    for m in range(anchors_6.shape[0]):
                        mean += [cx, cy, anchors_6[m, 0], anchors_6[m, 1]]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
