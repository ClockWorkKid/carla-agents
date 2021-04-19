import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import perception.tools._init_paths
import models
from config import config


class DDRNet:

    def __init__(self):
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        model_verbose = False

        if torch.__version__.startswith('1'):
            module = models.ddrnet_23_slim
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        self.model = models.ddrnet_23_slim.get_seg_model(config)

        model_state_file = 'perception/pretrained_models/best_val_smaller.pth'

        pretrained_dict = torch.load(model_state_file)
        model_dict = self.model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                           if k[6:] in model_dict.keys()}
        if model_verbose:
            for k, _ in pretrained_dict.items():
                print('=> loading {} from pretrained model'.format(k))
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        gpus = list((0,))
        self.model = nn.DataParallel(self.model, device_ids=gpus).cuda()

        self.model.eval()

        print("Segmentation Model Initialized!")

    def segments(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std

        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        with torch.no_grad():
            pred = self.model(torch.Tensor(image))

        return pred