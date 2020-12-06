'''
python test/demo_carplate_four_corners.py --input_size 512 --trained_model_bbox weights/CCPD_carplate_bbox_weights/ssd512_40000.pth --trained_model_only_four_corners weights/CCPD_carplate_only_four_corners_with_CIoU_loss_weights_16/ssd512_50000.pth --trained_model_four_corners weights/CCPD_carplate_bbox_four_corners_with_CIoU_loss_weights_16/ssd512_50000.pth
'''

import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
import sys
sys.path.append(".")
import argparse
import shapely
from shapely.geometry import Polygon, MultiPoint


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Testing With Pytorch')
parser.add_argument('--input_size', default=300, type=int, help='SSD300 or SSD512')
parser.add_argument('--trained_model_bbox',
                    default='weights/voc_weights/VOC300.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--trained_model_only_four_corners',
                    default='weights/voc_weights/VOC300.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--trained_model_four_corners',
                    default='weights/voc_weights/VOC300.pth', type=str,
                    help='Trained state_dict file path to open')
args = parser.parse_args()

from ssd import build_ssd
net_bbox = build_ssd('test', args.input_size, 2)    # initialize SSD
net_bbox.load_weights(args.trained_model_bbox)

from ssd_only_four_corners import build_ssd
net_only_four_corners = build_ssd('test', args.input_size, 2)    # initialize SSD
net_only_four_corners.load_weights(args.trained_model_only_four_corners)

from ssd_four_corners import build_ssd
net_four_corners = build_ssd('test', args.input_size, 2)    # initialize SSD
net_four_corners.load_weights(args.trained_model_four_corners)

# matplotlib inline
from matplotlib import pyplot as plt
from data import CARPLATE_FOUR_CORNERSDetection, CARPLATE_FOUR_CORNERSAnnotationTransform, CARPLATE_FOUR_CORNERS_ROOT
testset = CARPLATE_FOUR_CORNERSDetection(CARPLATE_FOUR_CORNERS_ROOT, None, None, CARPLATE_FOUR_CORNERSAnnotationTransform(),
                                         dataset_name='test')
for img_id in range(10053):
    if img_id not in [854,3661,6106,6227,6235]:
        continue
    image = testset.pull_image(img_id)
    _, target, _, _ = testset.pull_item(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    y_bbox = net_bbox(xx)
    y_only_four_corners = net_only_four_corners(xx)
    y_four_corners = net_four_corners(xx)

    from data import CARPLATE_FOUR_CORNERS_CLASSES as labels

    fig = plt.figure(figsize=(21, 7))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    # [num, num_classes, top_k, 13]
    # 13: score(1) bbox(4) corners(8)
    detections_bbox = y_bbox.data
    detections_only_four_corners = y_only_four_corners.data
    detections_four_corners = y_four_corners.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    scale_corner = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)

    for i in range(detections_bbox.size(1)):
        # skip background
        if i == 0:
            continue
        j = 0
        th = 0.5
        while detections_bbox[0, i, j, 0] > th:
            pt = (detections_bbox[0, i, j, 1:5] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i+2]

            ax_bbox = plt.subplot(1, 3, 1)
            ax_bbox.imshow(rgb_image)
            ax_bbox.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))

            only_four_corners = (detections_only_four_corners[0, i, j, 5:] * scale_corner).cpu().numpy()
            corners_x = np.append(only_four_corners[0::2], only_four_corners[0])
            corners_y = np.append(only_four_corners[1::2], only_four_corners[1])
            ax_only_four_corners = plt.subplot(1, 3, 3)
            ax_only_four_corners.imshow(rgb_image)
            ax_only_four_corners.plot(corners_x, corners_y, color=color, linewidth=2)

            four_corners = (detections_four_corners[0, i, j, 5:] * scale_corner).cpu().numpy()
            corners_x = np.append(four_corners[0::2], four_corners[0])
            corners_y = np.append(four_corners[1::2], four_corners[1])
            ax_four_corners = plt.subplot(1, 3, 2)
            ax_four_corners.imshow(rgb_image)
            ax_four_corners.plot(corners_x, corners_y, color=color, linewidth=2)

            # select possible examples
            only_four_corners_list = detections_only_four_corners[0, i, j, 5:].cpu().numpy().tolist()
            four_corners_list = detections_four_corners[0, i, j, 5:].cpu().numpy().tolist()
            iou_only_four_corners = polygon_iou(only_four_corners_list, target[0][4:12])
            iou_four_corners = polygon_iou(four_corners_list, target[0][4:12])
            if iou_only_four_corners - iou_four_corners > 0.1:
                print(img_id)

            j += 1

    plt.show()
