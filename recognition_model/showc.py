import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

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

if __name__ == '__main__':
    font_path = './simsun.ttc'
    img = np.random.randint(0, 255, [224, 224, 3], np.uint8)
    img = putText(img, "你好呀，年轻人", (0, 0), font_path, (0, 0, 255), 32)
    cv2.imshow('', img)
    cv2.waitKey()
