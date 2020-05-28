import torch

def parameter():
    batch_size = 100
    n_classes = 10
    epochs = 10

    image_width = 60
    learning_rate = 0.01

    """Regularized pooling is possible when kernel size is odd number."""
    pool_kernel = 5
    pool_stride = 5
    output_width = image_width // pool_kernel

    smooth_kernel = 3
    if smooth_kernel == 3:
        smooth_padding = 1
    elif smooth_kernel == 5:
        smooth_padding = 2

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    return batch_size, n_classes, epochs, image_width, learning_rate, pool_kernel, pool_stride, output_width, smooth_kernel, smooth_padding, device