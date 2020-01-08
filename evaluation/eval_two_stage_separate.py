"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import sys
sys.path.append(".")

from data import CAR_CARPLATEDetection, CAR_CARPLATEAnnotationTransform, CAR_CARPLATE_ROOT, BaseTransform
from data import CAR_CARPLATE_CLASSES as labelmap
import torch.utils.data as data

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model_1',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--trained_model_2',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=CAR_CARPLATE_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--input_size', default=300, type=int,
                    help='SSD300 OR SSD512')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'ImageSets',
                          'Main', '{:s}.txt')
devkit_path = args.voc_root
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, off_net, corners_net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_off': Timer(), 've_resize': Timer(), 'im_corners': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(save_folder + '/ssd' + str(args.input_size) + '_two_stage_separate', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_off'].tic()
        # [num, num_classes, top_k, 10]
        # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
        detections = off_net(x).data
        off_time = _t['im_off'].toc(average=False)

        # vehicle
        dets = detections[0, 1, :]
        mask = dets[:, 0].gt(0.).expand(10, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 10)
        if dets.size(0) == 0:
            continue
        boxes = dets[:, 1:]
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        scores = dets[:, 0].cpu().numpy()
        cls_dets = np.hstack((boxes.cpu().numpy(),
                                scores[:, np.newaxis])).astype(np.float32,
                                                                copy=False)
        all_boxes[1][i] = cls_dets

        _t['ve_resize'].tic()
        # license plate
        has_lp_th = 0.5
        th = 0.6
        scale = torch.Tensor([w, h])
        scale_2 = torch.Tensor([w, h]).repeat(2)

        # 让所有车都在图片范围内
        detections[:, :, :, 1:5] = torch.max(detections[:, :, :, 1:5], torch.zeros(detections[:, :, :, 1:5].shape))
        detections[:, :, :, 1:5] = torch.min(detections[:, :, :, 1:5], torch.ones(detections[:, :, :, 1:5].shape))

        # 同时有车并且车内有车牌
        has_car_idx = detections[0, 1, :, 0] > th
        has_lp_idx = detections[0, 1, :, 5] > has_lp_th
        has_car_lp_idx = has_car_idx * has_lp_idx
        # has_car_lp_idx可能为全0,直接跳过,不然会出bug
        if torch.sum(has_car_lp_idx).cpu().numpy() == 0:
            continue
        # car center
        car_pt = detections[0, 1, has_car_lp_idx, 1:5] * scale_2
        car_center = (car_pt[:, :2] + car_pt[:, 2:]) / 2
        # carplate center
        lp_size = detections[0, 1, has_car_lp_idx, 6:8] * scale
        lp_offset = detections[0, 1, has_car_lp_idx, 8:] * scale
        lp_center = car_center + lp_offset
        # 扩大车牌,并限制在车内
        expand_ratio = 3
        expanded_lp_top_left = lp_center - lp_size / 2 * expand_ratio
        expanded_lp_top_left = torch.max(expanded_lp_top_left, car_pt[:, :2])
        expanded_lp_bottom_right = lp_center + lp_size / 2 * expand_ratio
        expanded_lp_bottom_right = torch.min(expanded_lp_bottom_right, car_pt[:, 2:])
        expand_lp_tensor = torch.cat([expanded_lp_top_left, expanded_lp_bottom_right], 1)
        # 将车牌限制在图片内
        expand_lp_tensor = torch.max(expand_lp_tensor, torch.zeros(expand_lp_tensor.shape))
        img_border = scale.expand_as(expanded_lp_top_left).repeat(1, 2) - 1
        expand_lp_tensor = torch.min(expand_lp_tensor, img_border)
        # TODO: try to use crop and resize, resize expanded region
        expand_lp = expand_lp_tensor.cpu().numpy().astype(np.int)
        num = expand_lp.shape[0]
        total_xx = torch.zeros((num, 3, 300, 300))
        image = dataset.pull_image(i)
        for k in range(num):
            xmin = expand_lp[k, 0]
            ymin = expand_lp[k, 1]
            xmax = expand_lp[k, 2]
            ymax = expand_lp[k, 3]
            x = cv2.resize(image[ymin:ymax+1, xmin:xmax+1], (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = torch.autograd.Variable(x)  # wrap tensor in Variable
            total_xx[k, :, :, :] = xx
        resize_time = _t['ve_resize'].toc(average=False)

        if torch.cuda.is_available():
            total_xx = total_xx.cuda()

        # four corners forward
        # [num, num_classes, top_k, 13]
        # 13: score(1) bbox(4) corners(8)
        _t['im_corners'].tic()
        detections = corners_net(total_xx).data
        corners_time = _t['im_corners'].toc(average=False)

        # 每个区域只选取conf最大的车牌
        detection = detections[:, 1, 0, :]
        expand_size = expand_lp_tensor[:, 2:] - expand_lp_tensor[:, :2] + 1
        expand_top_left = expand_lp_tensor[:, :2]
        lp_box = detection[:, 1:5] * expand_size.repeat(1, 2) + expand_top_left.repeat(1, 2)
        scores = detection[:, 0].cpu().numpy()
        cls_dets = np.hstack((lp_box.cpu().numpy(),
                                scores[:, np.newaxis])).astype(np.float32,
                                                                copy=False)
        all_boxes[2][i] = cls_dets

        print('im_detect: {:d}/{:d} off:{:.4f} resize:{:.4f} corners:{:.4f} total:{:.4f}s'.format(i + 1,
                                                    num_images, off_time, resize_time, corners_time,
                                                    off_time + resize_time + corners_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap)                      # dont +1
    from ssd_offset import build_ssd
    off_net = build_ssd('test', args.input_size, num_classes)            # initialize SSD
    off_net.load_state_dict(torch.load(args.trained_model_1))
    off_net.eval()
    from ssd_four_corners import build_ssd
    corners_net = build_ssd('test', 300, num_classes)
    corners_net.load_state_dict(torch.load(args.trained_model_2))
    corners_net.eval()
    print('Finished loading model!')
    # load data
    dataset = CAR_CARPLATEDetection(root=args.voc_root,
                           transform=BaseTransform(args.input_size, dataset_mean),
                           target_transform=CAR_CARPLATEAnnotationTransform(keep_difficult=True),
                           dataset_name=set_type)
    if args.cuda:
        off_net = off_net.cuda()
        corners_net = corners_net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, off_net, corners_net, args.cuda, dataset,
             BaseTransform(off_net.size, dataset_mean), args.top_k, args.input_size,
             thresh=args.confidence_threshold)
