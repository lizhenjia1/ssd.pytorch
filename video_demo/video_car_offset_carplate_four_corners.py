from tkinter import *
import tkinter.filedialog

import torch
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from matplotlib import pyplot as plt

import sys
sys.path.append(".")

from data import CAR_CARPLATE_OFFSET_CLASSES as labels
# load model
from ssd_offset import build_ssd
offset_net = build_ssd('test', 300, 2)    # initialize SSD
offset_net.load_weights('weights/car_carplate_offset_weights/CAR_CARPLATE_OFFSET.pth')
from ssd_four_corners import build_ssd
corners_net = build_ssd('test', 300, 2)  # initialize SSD
corners_net.load_weights('weights/carplate_four_corners_with_border_weights/CARPLATE_FOUR_CORNERS_WITH_BORDER.pth')

from recognition_model import model_crnn_fc as crnn

# crnn params
crnn_model_path = 'recognition_model/crnn_best.pth'
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新领学使警挂港澳电'
nclass = len(alphabet) + 1

lp_rec_model = crnn.CRNN_FC(32, 1, nclass, 256)
lp_rec_model.load_state_dict(torch.load(crnn_model_path))


# crnn识别结果解码
class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()

        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


# crnn文本信息识别
def crnn_recognition(cropped_image, model):
    """Decode encoded texts back into strs.

        Args:
            numpy: a RGB license plate image
            model: <class 'models.crnn_fc.CRNN_FC'>

        Returns:
            str: predict result
    """

    imgH = 32
    imgW = 100
    mean = 0.588
    std = 0.193
    converter = strLabelConverter(alphabet)
    # print(alphabet)
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    ### ratio
    ### 280是中文训练集中图片的宽度，160是将原始图片缩小后的图片宽度
    w_now = int(image.shape[1] / (100 * 1.0 / imgW))
    h, w = image.shape
    image = cv2.resize(image, (0,0), fx=imgW/w, fy=imgH/h, interpolation=cv2.INTER_CUBIC)

    image = (np.reshape(image, (imgH, imgW, 1))).transpose(2, 0, 1)

    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(mean).div_(std)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = torch.autograd.Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = torch.autograd.Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))

    return sim_pred


from PIL import ImageFont, ImageDraw, Image

font_path = 'recognition_model/simsun.ttc'


def putText(img, text, org, font_path, color=(0, 0, 255), font_size=20):
    """
    在图片上显示文字
    :param img: 输入的img, 通过cv2读取
    :param text: 要显示的文字
    :param org: 文字左上角坐标
    :param font_path: 字体路径
    :param color: 字体颜色, (B,G,R)
    :return:
    """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    b, g, r = color
    a = 0
    draw.text(org, text, font=ImageFont.truetype(font_path, font_size), fill=(b, g, r, a))
    img = np.array(img_pil)
    return img


def video_run(dir_name):
    videoCapture = cv2.VideoCapture(dir_name)
    video_name = dir_name.strip().split('/')[-1].split('.')[0]
    video_suffix = dir_name.strip().split('/')[-1].split('.')[1]
    # from n-th frame
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    success, image = videoCapture.read()
    video = cv2.VideoWriter(video_name+'_result.'+video_suffix, cv2.VideoWriter_fourcc(*'XVID'), 25, size)

    cur_num = 0
    while success:
        image_copy = image.copy()
        img_h, img_w, _ = image.shape
        # skip frames
        if cur_num % 1 == 0:
            x = cv2.resize(image, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)

            xx = torch.autograd.Variable(x.unsqueeze(0))     # wrap tensor in Variable
            if torch.cuda.is_available():
                xx = xx.cuda()

            y = offset_net(xx)

            # [num, num_classes, top_k, 10]
            # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
            detections = y.data
            # scale each detection back up to the image
            scale = torch.Tensor(image.shape[1::-1]).repeat(2)
            scale_2 = torch.Tensor(image.shape[1::-1])
            scale_8 = torch.Tensor(image.shape[1::-1]).repeat(4)

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
                    cv2.rectangle(image_copy, (pt[0], pt[1]), (pt[2], pt[3]), (0, 0, 255), 2)
                    j += 1

                # has car and carplate
                has_car_idx = detections[0, i, :, 0] > th
                has_lp_idx = detections[0, i, :, 5] > th
                has_car_lp_idx = has_car_idx * has_lp_idx
                # has_car_lp_idx可能为全0,直接跳过,不过会出bug
                if torch.sum(has_car_lp_idx).cpu().numpy() == 0:
                    continue
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
                    x -= (104.0, 117.0, 123.0)
                    x = x.astype(np.float32)
                    x = x[:, :, ::-1].copy()
                    x = torch.from_numpy(x).permute(2, 0, 1)
                    xx = torch.autograd.Variable(x)  # wrap tensor in Variable
                    total_xx[k, :, :, :] = xx

                if torch.cuda.is_available():
                    total_xx = total_xx.cuda()

                # four corners forward
                y = corners_net(total_xx)

                # [num, num_classes, top_k, 13]
                # 13: score(1) bbox(4) corners(8)
                detections = y.data

                # 确保每个车牌都有对应的context,因为一辆车可能会检测到多个车牌
                lp_context_array = []
                lp_coor_array = []
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
                                # restricted in the image
                                lp_box = torch.max(lp_box, torch.zeros(lp_box.shape))
                                img_border = scale - 1
                                lp_box = torch.min(lp_box, img_border)
                                lp_box = lp_box.cpu().numpy()
                                # cv2.rectangle(image_copy, (lp_box[0], lp_box[1]), (lp_box[2], lp_box[3]), (0, 255, 0), 2)

                                lp_corners = detection[p, 5:] * (expand_size.repeat(4)) + expand_top_left.repeat(4)
                                # restricted in the image
                                lp_corners = torch.max(lp_corners, torch.zeros(lp_corners.shape))
                                img_border = scale_8 - 1
                                lp_corners = torch.min(lp_corners, img_border)
                                lp_corners = lp_corners.cpu().numpy()
                                corners_x = np.append(lp_corners[0::2], lp_corners[0])
                                corners_y = np.append(lp_corners[1::2], lp_corners[1])
                                for li in range(5):
                                    cv2.line(image_copy, (corners_x[li % 5], corners_y[li % 5]),
                                             (corners_x[(li+1) % 5], corners_y[(li+1) % 5]),
                                             (0, 255, 0), 2)
                                # 需要记录每个车牌的bbox和四点,以及对应车的位置
                                lp_context_array.append(car_pt[m, :2].cpu().numpy())
                                lp_coor_array.append(np.concatenate((lp_box, lp_corners)).tolist())

                # 根据bbox和四点做仿射变换,并且切出来对应的image
                lp_context_array = np.array(lp_context_array)
                lp_coor_array = np.array(lp_coor_array)
                for q in range(lp_coor_array.shape[0]):
                    carplate_xmin = lp_coor_array[q, 0]
                    carplate_ymin = lp_coor_array[q, 1]
                    carplate_xmax = lp_coor_array[q, 2]
                    carplate_ymax = lp_coor_array[q, 3]
                    pts1 = np.float32([[carplate_xmin, carplate_ymin],
                                       [carplate_xmax, carplate_ymin],
                                       [carplate_xmin, carplate_ymax]])

                    carplate_x_top_left = lp_coor_array[q, 4]
                    carplate_y_top_left = lp_coor_array[q, 5]
                    carplate_x_top_right = lp_coor_array[q, 6]
                    carplate_y_top_right = lp_coor_array[q, 7]
                    carplate_x_bottom_right = lp_coor_array[q, 8]
                    carplate_y_bottom_right = lp_coor_array[q, 9]
                    carplate_x_bottom_left = lp_coor_array[q, 10]
                    carplate_y_bottom_left = lp_coor_array[q, 11]
                    pts2 = np.float32([[carplate_x_top_left, carplate_y_top_left],
                                       [carplate_x_top_right, carplate_y_top_right],
                                       [carplate_x_bottom_left, carplate_y_bottom_left]
                                       ])

                    M = cv2.getAffineTransform(pts2, pts1)
                    dst = cv2.warpAffine(image, M, (img_w, img_h))
                    img_crop = dst[int(carplate_ymin):int(carplate_ymax)+1, int(carplate_xmin):int(carplate_xmax)+1]
                    # cv2.namedWindow("img_crop", 0)
                    # cv2.imshow('img_crop', img_crop)
                    # cv2.waitKey(0)

                    predict = crnn_recognition(img_crop, lp_rec_model)
                    image_copy = putText(image_copy, predict, (lp_context_array[q, 0], lp_context_array[q, 1]),
                                         font_path, (0, 255, 0), 30)

            plt.show()



        video.write(image_copy)
        cv2.imshow('image', image_copy)
        success, image = videoCapture.read()
        cur_num = cur_num + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    videoCapture.release()
    cv2.destroyAllWindows()


# select video from Dialog
root = Tk()


def xz():
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        lb.config(text="您选择的视频是："+filename)
        video_run(filename)
    else:
        lb.config(text="您没有选择任何视频")

lb = tkinter.Label(root, text='请选择视频文件')
lb.pack()
btn = tkinter.Button(root, text="弹出选择视频对话框", command=xz)
btn.pack()
root.mainloop()
