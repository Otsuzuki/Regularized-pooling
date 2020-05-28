import torch
import numpy as np
import torch.nn as nn
import scipy.stats as stats
from variable import parameter

batch_size, n_classes, epochs, image_width, learning_rate, pool_kernel, pool_stride, output_width, smooth_kernel, smooth_padding, device = parameter()

"""Get displacement features from max pooling"""
def displacement(indices, batch_size):
    channel = len(indices[1])

    # make some masks
    idx = torch.zeros([batch_size, channel, output_width, output_width]).to(device)
    x = torch.zeros([batch_size, channel, output_width, output_width]).to(device) 
    y = torch.zeros([batch_size, channel, output_width, output_width]).to(device)

    # Get the displacement features 
    for i in range(output_width):
        for j in range(output_width):
            idx[:, :, i, j] = i * image_width * pool_stride + j * pool_stride
            x[:, :, i, j] = (indices[:, :, i, j].float().to(device) - idx[:, :, i, j].float().to(device)) % image_width
            y[:, :, i, j] = (indices[:, :, i, j].float().to(device) - idx[:, :, i, j].float().to(device)) // image_width

    if (pool_kernel % 2) == 1: # when kernel size is odd number
        dis_x = x.to(device) - (pool_kernel - 1) / 2
        dis_y = -y.to(device) + (pool_kernel - 1) / 2
    elif (pool_kernel % 2) == 0: # when kernel size is even number
        numpy_x = x.numpy()
        numpy_y = y.numpy()
        dis_x = np.where(numpy_x < (pool_kernel / 2),numpy_x - pool_kernel / 2 , numpy_x - pool_kernel / 2 + 1)
        dis_y = np.where(numpy_y < (pool_kernel / 2),-numpy_y + pool_kernel / 2 , -numpy_y + (pool_kernel - 2) / 2 + 1)
    return dis_x, dis_y, x, y

""" If there is no max in kernel, displacement feature is 0 """
def dontcare(output, dis_x, dis_y, x, y, indices, batch_size):
    channel = len(indices[1])

    out1 = output.cpu().detach().numpy()
    np.set_printoptions(precision=5)
    out2 = out1.reshape(-1)
    out3,_= stats.mode(out2)
    A = out3[0]

    # make some masks
    mask_pool = torch.zeros([batch_size, channel, output_width, output_width]).to(device)
    mask_x = torch.zeros([batch_size, channel, output_width, output_width]).to(device)
    mask_y = torch.zeros([batch_size, channel, output_width, output_width]).to(device)
    mask_pool[output == A] = 1
    mask_x[x == 0] = 1
    mask_y[y == 0] = 1
    mask = mask_x + mask_y + mask_pool
    dis_x[mask == 3] = 0
    dis_y[mask == 3] = 0

    return dis_x, dis_y

""" Smoothing process to  displacement features in kernels """
def regularizeedprocess(indices_x_smooth, indices_y_smooth, indices, batch_size):
    channel = len(indices[1])

    # make some masks
    mask_x =torch.zeros((batch_size,channel,output_width ** 2)).to(device)
    mask_y =torch.zeros((batch_size,channel,output_width ** 2)).to(device)
    for i in range(channel):
        mask_x[:,i,:] = torch.sum(indices_x_smooth[:,range(i*(smooth_kernel**2),(i+1)*(smooth_kernel**2)),:], dim=1)
        mask_y[:,i,:] = torch.sum(indices_y_smooth[:,range(i*(smooth_kernel**2),(i+1)*(smooth_kernel**2)),:], dim=1)

    # smoothing process
    Avg_x = torch.round(mask_x / (smooth_kernel**2))
    Avg_y = torch.round(mask_y / (smooth_kernel**2))
    Avg_x_trans = Avg_x  + (pool_kernel - 1) / 2
    Avg_y_trans = -Avg_y + (pool_kernel - 1) / 2

    return Avg_x_trans, Avg_y_trans

""" Apply the smoothed result of the displacement feature to the original image. """
def apply(conv_unfold, Avg_x_trans, Avg_y_trans, indices, batch_size):
    channel = len(indices[1])

    conv_unfold = conv_unfold.reshape(batch_size,channel,pool_kernel,pool_kernel,output_width ** 2)
    conv_unfold = conv_unfold.permute(0,1,4,2,3)

    Avg_x_trans = Avg_x_trans.reshape(batch_size*channel*(output_width ** 2))
    Avg_y_trans = Avg_y_trans.reshape(batch_size*channel*(output_width ** 2))
    conv_unfold = conv_unfold.reshape(batch_size*channel*(output_width ** 2), pool_kernel, pool_kernel)

    Avg_x_trans = Avg_x_trans.long()
    Avg_y_trans = Avg_y_trans.long()

    unfold_pooled = conv_unfold[range(batch_size*channel*(output_width ** 2)), Avg_y_trans, Avg_x_trans]
    unfold_pooled = unfold_pooled.reshape(batch_size, channel, output_width ** 2)

    return unfold_pooled