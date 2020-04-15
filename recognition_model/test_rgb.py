import numpy as np
import sys, os
import time
import cv2
sys.path.append(os.getcwd())
# crnn packages
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import temp.crnn_fc_model as crnn
import Levenshtein

import params_qi as params
import alphabets

str1 = alphabets.alphabet
color_dict=['蓝','黄','绿','黑','白']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default="/rdata/qi.liu/code/LPR/ezai/all_projects/carplate_recognition/data/test_new/energy_longmao.txt", help='the path to your images')

opt = parser.parse_args()


# crnn params
crnn_model_path = '/rdata/qi.liu/code/LPR/pytorch_crnn/crnn_chinese_characters_rec-master/temp/model_text.pth'
alphabet = str1
nclass = len(alphabet)+1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# crnn文本信息识别,颜色识别
def crnn_recognition(image, model):
    imgH = 32
    imgW = 100
    converter = utils.strLabelConverter(alphabet)
    ### ratio
    h, w, c = image.shape

    image = cv2.resize(image, (0,0), fx=imgW/w, fy=imgH/h, interpolation=cv2.INTER_CUBIC)
    image = (np.reshape(image, (imgH, imgW, 3))).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)

    if torch.cuda.is_available():
        image = image.cuda(device)
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    print(preds.shape)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    log = converter.decode(preds.data, preds_size.data, raw=True)
    

    print("raw_data: ", log)
    return sim_pred\

if __name__ == '__main__':

    image_root = '/rdata/qi.liu/code/LPR/ezai/all_projects/carplate_recognition'
    f_wrong =open('results/wrong.txt', 'a')
	# crnn network 
    model = crnn.CRNN_FC(32, 3, 76, isPretrain=False, leakyRelu=False)

    if torch.cuda.is_available():
        model = model.cuda(device)
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))

    count_all = 0
    with open(opt.images_path, 'r' ,encoding='utf-8') as f:
        for item in f.readlines():
            print(count_all)
            count_all += 1
            image_path = os.path.join(image_root,item.strip().split(' ')[0])
            image_name = item.strip().split(' ')[0].split('/')[-1]
            label_text = item.strip().split(' ')[1]


            print(image_path)
            image = cv2.imread(image_path)
            pred_text = crnn_recognition(image, model)
            
            print('results: {0}'.format(pred_text), 'GT: {0}'.format(label_text) )

