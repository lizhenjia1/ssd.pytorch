# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, match_offset, match_four_corners, match_two_stage_end2end,\
    match_two_stage_end2end_offset, log_sum_exp
from ..box_utils import decode, decode_four_corners


class ProposalTargetLayer_offset(nn.Module):
    def __init__(self):
        super(ProposalTargetLayer_offset, self).__init__()

    def forward(self, rpn_rois, targets, expand_num):
        num = rpn_rois.shape[0]
        rois_t = []

        for idx in range(num):
            truths = targets[idx].data
            # TODO: 1需要修改,之后可扩展
            defaults = rpn_rois[idx, 1, :, :]
            match_two_stage_end2end_offset(truths, defaults, rois_t, expand_num)

        return rois_t