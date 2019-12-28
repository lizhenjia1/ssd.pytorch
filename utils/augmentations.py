import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToAbsoluteCoords_offset(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        boxes[:, 5] *= width
        boxes[:, 7] *= width
        boxes[:, 6] *= height
        boxes[:, 8] *= height

        return image, boxes, labels


class ToAbsoluteCoords_four_corners(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        boxes[:, 4] *= width
        boxes[:, 6] *= width
        boxes[:, 5] *= height
        boxes[:, 7] *= height
        boxes[:, 8] *= width
        boxes[:, 10] *= width
        boxes[:, 9] *= height
        boxes[:, 11] *= height

        return image, boxes, labels


class ToAbsoluteCoords_two_stage_end2end(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        boxes[:, 5] *= width
        boxes[:, 7] *= width
        boxes[:, 6] *= height
        boxes[:, 8] *= height
        boxes[:, 9] *= width
        boxes[:, 11] *= width
        boxes[:, 10] *= height
        boxes[:, 12] *= height
        boxes[:, 13] *= width
        boxes[:, 15] *= width
        boxes[:, 14] *= height
        boxes[:, 16] *= height
        boxes[:, 17] *= width
        boxes[:, 19] *= width
        boxes[:, 18] *= height
        boxes[:, 20] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class ToPercentCoords_offset(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        boxes[:, 5] /= width
        boxes[:, 7] /= width
        boxes[:, 6] /= height
        boxes[:, 8] /= height

        return image, boxes, labels


class ToPercentCoords_four_corners(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        boxes[:, 4] /= width
        boxes[:, 6] /= width
        boxes[:, 5] /= height
        boxes[:, 7] /= height
        boxes[:, 8] /= width
        boxes[:, 10] /= width
        boxes[:, 9] /= height
        boxes[:, 11] /= height

        return image, boxes, labels


class ToPercentCoords_two_stage_end2end(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        boxes[:, 5] /= width
        boxes[:, 7] /= width
        boxes[:, 6] /= height
        boxes[:, 8] /= height
        boxes[:, 9] /= width
        boxes[:, 11] /= width
        boxes[:, 10] /= height
        boxes[:, 12] /= height
        boxes[:, 13] /= width
        boxes[:, 15] /= width
        boxes[:, 14] /= height
        boxes[:, 16] /= height
        boxes[:, 17] /= width
        boxes[:, 19] /= width
        boxes[:, 18] /= height
        boxes[:, 20] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                # if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


# 一旦切到了车牌就continue
class RandomSampleCrop_offset(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                # size of original image [0.09, 1)
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:4] = np.minimum(current_boxes[:, 2:4], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:4] -= rect[:2]

                # has carplate and whole of carplate is in batch
                has_carplate_mask = current_boxes[:, 4].astype(np.int32) > 0

                carplate_centers = centers[mask, :] + current_boxes[:, 7:9]

                # 这个是车牌中心在crop中的情况
                # carplate_m1 = (rect[0] < carplate_centers[:, 0]) * (rect[1] < carplate_centers[:, 1])
                # carplate_m2 = (rect[2] > carplate_centers[:, 0]) * (rect[3] > carplate_centers[:, 1])
                # carplate_mask = has_carplate_mask * carplate_m1 * carplate_m2

                # carplate left_top and right_bottom
                carplate_xmin = (carplate_centers[:, 0] - current_boxes[:, 5] / 2).reshape((-1, 1))
                carplate_ymin = (carplate_centers[:, 1] - current_boxes[:, 6] / 2).reshape((-1, 1))
                carplate_xmax = (carplate_centers[:, 0] + current_boxes[:, 5] / 2).reshape((-1, 1))
                carplate_ymax = (carplate_centers[:, 1] + current_boxes[:, 6] / 2).reshape((-1, 1))

                carplate_bbox = np.hstack((carplate_xmin, carplate_ymin, carplate_xmax, carplate_ymax))
                carplate_overlap = jaccard_numpy(carplate_bbox, rect)
                carplate_m1 = carplate_overlap < 1e-14  # 车牌完全在crop外
                area_carplate = (carplate_xmax - carplate_xmin) * (carplate_ymax - carplate_ymin)
                area_carplate = area_carplate.reshape((-1))  # 这里很关键,不然是一个矩阵
                carplate_m2 = abs(intersect(carplate_bbox, rect) - area_carplate) < 1e-14  # 车牌完全在crop内
                carplate_not_crop_mask = carplate_m1 | carplate_m2

                # 如果有车牌被切到直接跳过
                if not carplate_not_crop_mask.all():
                    continue

                # 有车牌并且车牌全部在crop中才继续
                carplate_mask = has_carplate_mask * carplate_m2

                # new position of carplate
                carplate_xmin = np.maximum(carplate_xmin, rect[0])
                carplate_xmin -= rect[0]
                carplate_ymin = np.maximum(carplate_ymin, rect[1])
                carplate_ymin -= rect[1]
                carplate_xmax = np.minimum(carplate_xmax, rect[2])
                carplate_xmax -= rect[0]
                carplate_ymax = np.minimum(carplate_ymax, rect[3])
                carplate_ymax -= rect[1]

                # new size and offset of carplate
                current_boxes[carplate_mask, 5] = carplate_xmax[carplate_mask, 0] - carplate_xmin[carplate_mask, 0]
                current_boxes[carplate_mask, 6] = carplate_ymax[carplate_mask, 0] - carplate_ymin[carplate_mask, 0]
                centers_new = (current_boxes[carplate_mask, :2] + current_boxes[carplate_mask, 2:4]) / 2.0
                carplate_centers_new = (np.hstack((carplate_xmin[carplate_mask], carplate_ymin[carplate_mask])) +
                                        np.hstack((carplate_xmax[carplate_mask], carplate_ymax[carplate_mask]))) / 2.0
                current_boxes[carplate_mask, 7:9] = carplate_centers_new - centers_new

                # without carplate, set to 0
                current_boxes[~carplate_mask, 4:] = 0

                return current_image, current_boxes, current_labels


# 一旦切到了车牌就continue
class RandomSampleCrop_two_stage_end2end(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                # size of original image [0.09, 1)
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:4] = np.minimum(current_boxes[:, 2:4], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:4] -= rect[:2]

                # has carplate and whole of carplate is in batch
                has_carplate_mask = current_boxes[:, 4].astype(np.int32) > 0

                # 这个是车牌中心在crop中的情况
                # carplate_m1 = (rect[0] < carplate_centers[:, 0]) * (rect[1] < carplate_centers[:, 1])
                # carplate_m2 = (rect[2] > carplate_centers[:, 0]) * (rect[3] > carplate_centers[:, 1])
                # carplate_mask = has_carplate_mask * carplate_m1 * carplate_m2

                # carplate left_top and right_bottom
                carplate_bbox = current_boxes[:, 9:13]
                carplate_xmin = carplate_bbox[:, 0].reshape((-1, 1))
                carplate_ymin = carplate_bbox[:, 1].reshape((-1, 1))
                carplate_xmax = carplate_bbox[:, 2].reshape((-1, 1))
                carplate_ymax = carplate_bbox[:, 3].reshape((-1, 1))

                # carplate four points
                carplate_four_points = current_boxes[:, 13:21]
                carplate_x_top_left = carplate_four_points[:, 0].reshape((-1, 1))
                carplate_y_top_left = carplate_four_points[:, 1].reshape((-1, 1))
                carplate_x_top_right = carplate_four_points[:, 2].reshape((-1, 1))
                carplate_y_top_right = carplate_four_points[:, 3].reshape((-1, 1))
                carplate_x_bottom_right = carplate_four_points[:, 4].reshape((-1, 1))
                carplate_y_bottom_right = carplate_four_points[:, 5].reshape((-1, 1))
                carplate_x_bottom_left = carplate_four_points[:, 6].reshape((-1, 1))
                carplate_y_bottom_left = carplate_four_points[:, 7].reshape((-1, 1))

                carplate_overlap = jaccard_numpy(carplate_bbox, rect)
                carplate_m1 = carplate_overlap < 1e-14  # 车牌完全在crop外
                area_carplate = (carplate_xmax - carplate_xmin) * (carplate_ymax - carplate_ymin)
                area_carplate = area_carplate.reshape((-1))  # 这里很关键,不然是一个矩阵
                carplate_m2 = abs(intersect(carplate_bbox, rect) - area_carplate) < 1e-14  # 车牌完全在crop内
                carplate_not_crop_mask = carplate_m1 | carplate_m2

                # 如果有车牌被切到直接跳过
                if not carplate_not_crop_mask.all():
                    continue

                # 有车牌并且车牌全部在crop中才继续
                carplate_mask = has_carplate_mask * carplate_m2

                # new position of carplate
                carplate_xmin = np.maximum(carplate_xmin, rect[0])
                carplate_xmin -= rect[0]
                carplate_ymin = np.maximum(carplate_ymin, rect[1])
                carplate_ymin -= rect[1]
                carplate_xmax = np.minimum(carplate_xmax, rect[2])
                carplate_xmax -= rect[0]
                carplate_ymax = np.minimum(carplate_ymax, rect[3])
                carplate_ymax -= rect[1]

                # new four points of carplate
                carplate_x_top_left -= rect[0]
                carplate_y_top_left -= rect[1]
                carplate_x_top_right -= rect[0]
                carplate_y_top_right -= rect[1]
                carplate_x_bottom_right -= rect[0]
                carplate_y_bottom_right -= rect[1]
                carplate_x_bottom_left -= rect[0]
                carplate_y_bottom_left -= rect[1]

                # new size and offset of carplate
                current_boxes[carplate_mask, 5] = carplate_xmax[carplate_mask, 0] - carplate_xmin[carplate_mask, 0]
                current_boxes[carplate_mask, 6] = carplate_ymax[carplate_mask, 0] - carplate_ymin[carplate_mask, 0]
                centers_new = (current_boxes[carplate_mask, :2] + current_boxes[carplate_mask, 2:4]) / 2.0
                carplate_centers_new = (np.hstack((carplate_xmin[carplate_mask], carplate_ymin[carplate_mask])) +
                                        np.hstack((carplate_xmax[carplate_mask], carplate_ymax[carplate_mask]))) / 2.0

                current_boxes[carplate_mask, 7:9] = carplate_centers_new - centers_new

                current_boxes[carplate_mask, 9] = carplate_xmin[carplate_mask, 0]
                current_boxes[carplate_mask, 10] = carplate_ymin[carplate_mask, 0]
                current_boxes[carplate_mask, 11] = carplate_xmax[carplate_mask, 0]
                current_boxes[carplate_mask, 12] = carplate_ymax[carplate_mask, 0]

                current_boxes[carplate_mask, 13] = carplate_x_top_left[carplate_mask, 0]
                current_boxes[carplate_mask, 14] = carplate_y_top_left[carplate_mask, 0]
                current_boxes[carplate_mask, 15] = carplate_x_top_right[carplate_mask, 0]
                current_boxes[carplate_mask, 16] = carplate_y_top_right[carplate_mask, 0]
                current_boxes[carplate_mask, 17] = carplate_x_bottom_right[carplate_mask, 0]
                current_boxes[carplate_mask, 18] = carplate_y_bottom_right[carplate_mask, 0]
                current_boxes[carplate_mask, 19] = carplate_x_bottom_left[carplate_mask, 0]
                current_boxes[carplate_mask, 20] = carplate_y_bottom_left[carplate_mask, 0]

                # without carplate, set to 0
                current_boxes[~carplate_mask, 4:] = 0

                return current_image, current_boxes, current_labels


# 作用就是在图片周围扩出一些区域,原图内容并不变,扩张后的图片有所变化,之后再resize;因为offset和size都不变,所以该函数适用于offset
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):  # 50% 概率expand
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:4] += (int(left), int(top))

        return image, boxes, labels


class Expand_four_corners(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):  # 50% 概率expand
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:4] += (int(left), int(top))
        boxes[:, 4:6] += (int(left), int(top))
        boxes[:, 6:8] += (int(left), int(top))
        boxes[:, 8:10] += (int(left), int(top))
        boxes[:, 10:12] += (int(left), int(top))

        return image, boxes, labels


class Expand_two_stage_end2end(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):  # 50% 概率expand
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:4] += (int(left), int(top))
        boxes[:, 9:11] += (int(left), int(top))
        boxes[:, 11:13] += (int(left), int(top))
        boxes[:, 13:15] += (int(left), int(top))
        boxes[:, 15:17] += (int(left), int(top))
        boxes[:, 17:19] += (int(left), int(top))
        boxes[:, 19:21] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


# 所谓镜像就是坐标x变为(width - x)
class RandomMirror_offset(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]  # 实现图片倒序;start:end:stride,当stride<0时,倒序输出,并且当start和end都缺省时从-1开始,此时行不变
            boxes = boxes.copy()
            boxes[:, 0:4:2] = width - boxes[:, 2::-2]  # xmax变xmin, xmin变xmax
            boxes[:, 7] = -boxes[:, 7]  # x_offset取相反数
        return image, boxes, classes


class RandomMirror_four_corners(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]  # 实现图片倒序;start:end:stride,当stride<0时,倒序输出,并且当start和end都缺省时从-1开始,此时行不变
            boxes = boxes.copy()
            boxes[:, 0:4:2] = width - boxes[:, 2::-2]  # xmax变xmin, xmin变xmax
            # 四边形上面两点交换
            boxes[:, 4:8:2] = width - boxes[:, 6:3:-2]
            boxes[:, [5, 7]] = boxes[:, [7, 5]]
            # 四边形下面两点交换
            boxes[:, 8:12:2] = width - boxes[:, 10:7:-2]
            boxes[:, [9, 11]] = boxes[:, [11, 9]]
        return image, boxes, classes


# 所谓镜像就是坐标x变为(width - x)
class RandomMirror_two_stage_end2end(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]  # 实现图片倒序;start:end:stride,当stride<0时,倒序输出,并且当start和end都缺省时从-1开始,此时行不变
            boxes = boxes.copy()
            boxes[:, 0:4:2] = width - boxes[:, 2::-2]  # xmax变xmin, xmin变xmax
            boxes[:, 7] = -boxes[:, 7]  # x_offset取相反数

            boxes[:, 9:13:2] = width - boxes[:, 11:8:-2]
            # 四边形上面两点交换
            boxes[:, 13:17:2] = width - boxes[:, 15:12:-2]
            boxes[:, [14, 16]] = boxes[:, [16, 14]]
            # 四边形下面两点交换
            boxes[:, 17:21:2] = width - boxes[:, 19:16:-2]
            boxes[:, [18, 20]] = boxes[:, [20, 18]]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class SSDAugmentation_offset(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords_offset(),
            # PhotometricDistort(),
            # Expand(self.mean),
            # RandomSampleCrop_offset(),
            # RandomMirror_offset(),
            ToPercentCoords_offset(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class SSDAugmentation_four_corners(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords_four_corners(),
            # PhotometricDistort(),
            # Expand_four_corners(self.mean),
            # RandomMirror_four_corners(),
            ToPercentCoords_four_corners(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class SSDAugmentation_two_stage_end2end(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords_two_stage_end2end(),
            PhotometricDistort(),
            Expand_two_stage_end2end(self.mean),
            RandomSampleCrop_two_stage_end2end(),
            RandomMirror_two_stage_end2end(),
            ToPercentCoords_two_stage_end2end(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
