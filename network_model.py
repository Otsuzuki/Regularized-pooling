import torch.nn as nn
import torch.nn.functional as F
from pooling_layer import Regularizedpooling
from variable import parameter

batch_size, n_classes, epochs, image_width, learning_rate, pool_kernel, pool_stride, output_width, smooth_kernel, smooth_padding, device = parameter()
class VGG11(nn.Module):
    def __init__(self, n_classes):
        super(VGG11, self).__init__()

        self.block1_output = nn.Sequential(
            Regularizedpooling(),
        )
        self.block2_output = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3_output = nn.Sequential (
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        x = self.block1_output(x)
        x = self.block2_output(x)
        x = self.block3_output(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x      