from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox_2(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox_2, self).__init__()
        self.image_size = cfg['min_dim_2']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios_2'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps_2']  # 不会用到真实特征图的大小，这里只是用来确定有几个特征图用来出head网络
        self.min_sizes = cfg['min_sizes_2']
        self.max_sizes = cfg['max_sizes_2']
        self.steps = cfg['steps_2']
        self.aspect_ratios = cfg['aspect_ratios_2']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  # 笛卡尔积，repeat=2表示与自身笛卡尔积
                f_k = self.feature_maps[k]  # 近似特征图大小
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size  # 每一个anchor都是以原图为基准，缩放到[0,1]，越往后anchor越大
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)

        return output
