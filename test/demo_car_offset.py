import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import sys
sys.path.append(".")

from ssd_offset import build_ssd

net = build_ssd('test', 300, 2)    # initialize SSD
net.load_weights('weights/car_carplate_offset_weights/CAR_CARPLATE_OFFSET.pth')

# matplotlib inline
from matplotlib import pyplot as plt
from data import CAR_CARPLATE_OFFSETDetection, CAR_CARPLATE_OFFSETAnnotationTransform, CAR_CARPLATE_OFFSET_ROOT
testset = CAR_CARPLATE_OFFSETDetection(CAR_CARPLATE_OFFSET_ROOT, None, None, CAR_CARPLATE_OFFSETAnnotationTransform(),
                                       dataset_name='test')
for img_id in range(80):
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

    y = net(xx)
    print(y.shape)

    from data import CAR_CARPLATE_CLASSES as labels

    fig = plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # [num, num_classes, top_k, 10]
    # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
    detections = y.data
    print(detections.shape)
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        # skip background
        if i == 0:
            continue
        j = 0
        th = 0.5
        while detections[0, i, j, 0] > th:
            score = detections[0, i, j, 0]
            has_lp = detections[0, i, j, 5]
            label_name = labels[i-1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})

            if has_lp > th:
                size_lp_offset = (detections[0, i, j, 6:] * scale).cpu().numpy()
                size_lp = size_lp_offset[:2]
                size_lp = abs(size_lp)
                offset = size_lp_offset[2:]
                center = ((pt[0] + pt[2]) / 2, (pt[1] + pt[3]) / 2)
                center_lp = center + offset
                currentAxis.plot(center[0], center[1], 'o')
                currentAxis.plot((center[0], center_lp[0]), (center[1], center_lp[1]))

                coords_lp = center_lp - size_lp / 2, size_lp[0], size_lp[1]
                currentAxis.add_patch(plt.Rectangle(*coords_lp, fill=False, edgecolor=color, linewidth=2))

            j += 1
    plt.show()
