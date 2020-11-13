import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import sys
sys.path.append(".")

from ssd_offset import build_ssd

offset_net = build_ssd('test', 300, 2)    # initialize SSD
offset_net.load_weights('weights/car_carplate_offset_weights/CAR_CARPLATE_OFFSET.pth')

from ssd_four_corners import build_ssd

corners_net = build_ssd('test', 300, 2)  # initialize SSD
corners_net.load_weights('weights/carplate_four_corners_with_border_weights/CARPLATE_FOUR_CORNERS_WITH_BORDER.pth')

# matplotlib inline
from matplotlib import pyplot as plt
from data import CAR_CARPLATE_OFFSETDetection, CAR_CARPLATE_OFFSETAnnotationTransform, CAR_CARPLATE_OFFSET_ROOT
testset = CAR_CARPLATE_OFFSETDetection(CAR_CARPLATE_OFFSET_ROOT, None, None, CAR_CARPLATE_OFFSETAnnotationTransform(),
                                       dataset_name = 'test')
for img_id in range(45):
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    y = offset_net(xx)

    from data import CAR_CARPLATE_OFFSET_CLASSES as labels

    fig = plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # [num, num_classes, top_k, 10]
    # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    scale_2 = torch.Tensor(rgb_image.shape[1::-1])

    # 让所有车都在图片范围内
    detections[:, :, :, 1:5] = torch.max(detections[:, :, :, 1:5], torch.zeros(detections[:, :, :, 1:5].shape))
    detections[:, :, :, 1:5] = torch.min(detections[:, :, :, 1:5], torch.ones(detections[:, :, :, 1:5].shape))
    for i in range(detections.size(1)):
        # skip background
        if i == 0:
            continue
        j = 0
        th = 0.6
        while detections[0, i, j, 0] > th:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:5] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=colors[0], linewidth=2))
            # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j += 1

        # has car and carplate
        has_car_idx = detections[0, i, :, 0] > th
        has_lp_idx = detections[0, i, :, 5] > th
        has_car_lp_idx = has_car_idx * has_lp_idx
        # car center
        car_pt = detections[0, i, has_car_lp_idx, 1:5] * scale
        car_center = (car_pt[:, :2] + car_pt[:, 2:]) / 2
        # carplate center
        lp_size = detections[0, i, has_car_lp_idx, 6:8] * scale_2
        lp_offset = detections[0, i, has_car_lp_idx, 8:] * scale_2
        lp_center = car_center + lp_offset
        # expand carplate
        expand_ratio = 3
        expanded_lp_top_left = lp_center - lp_size / 2 * expand_ratio
        expanded_lp_top_left = torch.max(expanded_lp_top_left, car_pt[:, :2])
        expanded_lp_bottom_right = lp_center + lp_size / 2 * expand_ratio
        expanded_lp_bottom_right = torch.min(expanded_lp_bottom_right, car_pt[:, 2:])
        expand_lp_tensor = torch.cat([expanded_lp_top_left, expanded_lp_bottom_right], 1)
        # restricted in the image
        expand_lp_tensor = torch.max(expand_lp_tensor, torch.zeros(expand_lp_tensor.shape))
        img_border = scale_2.expand_as(expanded_lp_top_left).repeat(1, 2) - 1
        expand_lp_tensor = torch.min(expand_lp_tensor, img_border)
        # resize expanded region
        expand_lp = expand_lp_tensor.cpu().numpy().astype(np.int)
        num = expand_lp.shape[0]
        total_xx = torch.zeros((num, 3, 300, 300))
        for k in range(num):
            xmin = expand_lp[k, 0]
            ymin = expand_lp[k, 1]
            xmax = expand_lp[k, 2]
            ymax = expand_lp[k, 3]
            x = cv2.resize(image[ymin:ymax+1, xmin:xmax+1], (300, 300)).astype(np.float32)
            # cv2.namedWindow("image", 0)
            # cv2.imshow('image', image[ymin:ymax+1, xmin:xmax+1])
            # cv2.waitKey(0)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = Variable(x)  # wrap tensor in Variable
            total_xx[k, :, :, :] = xx

        if torch.cuda.is_available():
            total_xx = total_xx.cuda()

        # four corners forward
        y = corners_net(total_xx)

        # [num, num_classes, top_k, 13]
        # 13: score(1) bbox(4) corners(8)
        detections = y.data

        for m in range(num):
            for n in range(detections.size(1)):
                th = 0.7
                # skip background
                if n == 0:
                    continue
                # 有可能出现多个车牌大于th
                has_lp_corners_idx = detections[m, n, :, 0] > th
                detection = detections[m, n, has_lp_corners_idx, :]
                # 去掉粗定位觉得有车牌,但是精定位没有车牌的情况;跟这里的th没有关系,直接是没有detection结果的
                if detection.shape[0] > 0:
                    expand_size = expand_lp_tensor[m, 2:] - expand_lp_tensor[m, :2] + 1
                    expand_top_left = expand_lp_tensor[m, :2]
                    # 可能多个车牌
                    for p in range(detection.shape[0]):
                        lp_box = detection[p, 1:5] * (expand_size.repeat(2)) + expand_top_left.repeat(2)
                        lp_box = lp_box.cpu().numpy()
                        # coords_lp = lp_box[:2], lp_box[2] - lp_box[0] + 1, lp_box[3] - lp_box[1] + 1
                        # currentAxis.add_patch(plt.Rectangle(*coords_lp, fill=False, edgecolor=colors[10], linewidth=2))

                        lp_corners = detection[p, 5:] * (expand_size.repeat(4)) + expand_top_left.repeat(4)
                        lp_corners = lp_corners.cpu().numpy()
                        corners_x = np.append(lp_corners[0::2], lp_corners[0])
                        corners_y = np.append(lp_corners[1::2], lp_corners[1])
                        plt.plot(corners_x, corners_y, color=colors[10])

    plt.show()
