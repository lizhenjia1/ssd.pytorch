import torch
from torch.autograd import Variable
import numpy as np
import cv2
import os
import sys
sys.path.append(".")
from ssd_four_corners import build_ssd
import argparse


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')





parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Testing With Pytorch')
parser.add_argument('--input_size', default=300, type=int, help='SSD300 or SSD512')
parser.add_argument('--trained_model',
                    default='weights/ccpd_300/ssd300_44000.pth', type=str,
                    help='Trained state_dict file path to open')
args = parser.parse_args()


net = build_ssd('test', args.input_size, 2)    # initialize SSD
net.load_weights(args.trained_model)

root_path = 'ssd_result/IMAGE'
save_path = 'ssd_result/IMAGE_1110_re'





for filename in os.listdir(root_path):
    if not filename.endswith('.jpg'):
        continue
    file = os.path.join(root_path,filename)
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    y = net(xx)

    labels = ['carplate']

    # [num, num_classes, top_k, 13]
    # 13: score(1) bbox(4) corners(8)
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    scale_corner = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)
    for i in range(detections.size(1)):
        if i==0:
            continue
        j = 0
        th = 0.3
        while detections[0, i, j, 0] > th:
            score = detections[0, i, j, 0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f' % (label_name, score)
            
            pt = (detections[0, i, j, 1:5] * scale).cpu().numpy()

            # coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1

            xmin = pt[0]
            ymin = pt[1]
            xmax = pt[2]
            ymax = pt[3]

            image = cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0, 0, 0),3)

            four_corners = (detections[0, i, j, 5:] * scale_corner).cpu().numpy()
            corners_x = np.append(four_corners[0::2], four_corners[0])
            corners_y = np.append(four_corners[1::2], four_corners[1])

            x_top_left = four_corners[0]
            y_top_left = four_corners[1]
            x_top_right = four_corners[2]
            y_top_right = four_corners[3]   
            x_bottom_right = four_corners[4]
            y_bottom_right = four_corners[5]
            x_bottom_left = four_corners[6]
            y_bottom_left = four_corners[7]

            image = cv2.line(image,(x_top_left,y_top_left),(x_top_right,y_top_right),(0, 255, 255),3)

            image = cv2.line(image,(x_top_left,y_top_left),(x_bottom_left,y_bottom_left),(255, 0, 0),3)

            image = cv2.line(image,(x_bottom_right,y_bottom_right),(x_top_right,y_top_right),(255, 255, 0),3)

            image = cv2.line(image,(x_bottom_right,y_bottom_right),(x_bottom_left,y_bottom_left),(0, 0, 255),3)


            j += 1
    cv2.imshow('image',image)
    cv2.waitKey(1000)
    # cv2.imwrite(save_path+'/'+filename,image)