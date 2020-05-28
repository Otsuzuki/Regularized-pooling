import torch
import torch.nn as nn
import torch.nn.functional as F
from  regularize import displacement, dontcare, regularizeedprocess, apply
from variable import parameter

batch_size, n_classes, epochs, image_width, learning_rate, pool_kernel, pool_stride, output_width, smooth_kernel, smooth_padding, device = parameter()

class Regularizedpooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential (
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential (
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel, return_indices=True),
        )
        self.unfold = nn.Sequential(
            nn.Unfold(kernel_size=pool_kernel, stride=pool_kernel)
        )
        self.slidewindow = nn.Sequential(
            nn.Unfold(kernel_size=smooth_kernel, padding=smooth_padding, stride=1)
        )
        self.fold = nn.Fold(output_size=(output_width, output_width), kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        input = self.conv(input)
        output, indices = self.pool(input)
        
        """Get displacement features from max pooling"""
        dis_x, dis_y, x, y = displacement(indices, batch_size)
        
        """ If there is no max in kernel, displacement feature is 0 """
        dis_x, dis_y = dontcare(output, dis_x, dis_y, x, y, indices, batch_size)

        indices_x_smooth = self.slidewindow(dis_x).to(device)
        indices_y_smooth = self.slidewindow(dis_y).to(device)

        """ Smoothing process to  displacement features in kernels """
        Avg_x_trans, Avg_y_trans = regularizeedprocess(indices_x_smooth, indices_y_smooth, indices, batch_size)

        conv_unfold = self.unfold(input)
        
        """ Apply the smoothed result of the displacement feature to the original image. """
        unfold_pooled = apply(conv_unfold, Avg_x_trans, Avg_y_trans, indices,batch_size)

        fold_out = self.fold(unfold_pooled)

        return fold_out
