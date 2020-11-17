import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import car, carplate, change_cfg_for_ssd512
import os


class SSD_two_branch(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, car_head, carplate_head, num_classes):
        super(SSD_two_branch, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.car_cfg = car
        self.carplate_cfg = carplate
        if size == 512:
            self.car_cfg = change_cfg_for_ssd512(self.car_cfg)
            self.carplate_cfg = change_cfg_for_ssd512(self.carplate_cfg)
        self.car_priorbox = PriorBox(self.car_cfg)
        self.carplate_priorbox = PriorBox(self.carplate_cfg)
        with torch.no_grad():
            self.car_priors = Variable(self.car_priorbox.forward())
        with torch.no_grad():
            self.carplate_priors = Variable(self.carplate_priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.car_loc = nn.ModuleList(car_head[0])
        self.car_conf = nn.ModuleList(car_head[1])

        self.carplate_loc = nn.ModuleList(carplate_head[0])
        self.carplate_conf = nn.ModuleList(carplate_head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        car_sources = list()
        car_loc = list()
        car_conf = list()

        carplate_sources = list()
        carplate_loc = list()
        carplate_conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        car_sources.append(s)
        carplate_sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        car_sources.append(x)
        carplate_sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                car_sources.append(x)
                carplate_sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(car_sources, self.car_loc, self.car_conf):
            car_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            car_conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # apply multibox head to source layers
        for (x, l, c) in zip(carplate_sources, self.carplate_loc, self.carplate_conf):
            carplate_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            carplate_conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        car_loc = torch.cat([o.view(o.size(0), -1) for o in car_loc], 1)
        car_conf = torch.cat([o.view(o.size(0), -1) for o in car_conf], 1)

        carplate_loc = torch.cat([o.view(o.size(0), -1) for o in carplate_loc], 1)
        carplate_conf = torch.cat([o.view(o.size(0), -1) for o in carplate_conf], 1)

        if self.phase == "test":
            output1 = self.detect(
                car_loc.view(car_loc.size(0), -1, 4),           # loc preds
                self.softmax(car_conf.view(car_conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.car_priors.type(type(x.data)),             # default boxes
            )
            output2 = self.detect(
                carplate_loc.view(carplate_loc.size(0), -1, 4), # loc preds
                self.softmax(carplate_conf.view(carplate_conf.size(0), -1,
                                           self.num_classes)),  # conf preds
                self.carplate_priors.type(type(x.data))         # default boxes
            )
            output = torch.cat((output1, output2[:,1,:,:].unsqueeze(1)), 1)
        else:
            output = (
                car_loc.view(car_loc.size(0), -1, 4),
                car_conf.view(car_conf.size(0), -1, self.num_classes),
                self.car_priors,
                carplate_loc.view(carplate_loc.size(0), -1, 4),
                carplate_conf.view(carplate_conf.size(0), -1, self.num_classes),
                self.carplate_priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # SSD512 need add two more Conv layer
    if size == 512:
        layers += [nn.Conv2d(in_channels, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    car_loc_layers = []
    car_conf_layers = []
    carplate_loc_layers = []
    carplate_conf_layers = []
    car_vgg_source = [21, -2]
    carplate_vgg_source = [21, -2]
    for k, v in enumerate(car_vgg_source):
        car_loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        car_conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(carplate_vgg_source):
        carplate_loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        carplate_conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        car_loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        car_conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        carplate_loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        carplate_conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    
    return vgg, extra_layers, (car_loc_layers, car_conf_layers), (carplate_loc_layers, carplate_conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 SSD512 (size=300 or size=512) is supported!")
        return
    base_, extras_, car_head_, carplate_head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], size, 1024),
                                     mbox[str(size)], num_classes)
    return SSD_two_branch(phase, size, base_, extras_, car_head_, carplate_head_, num_classes)
