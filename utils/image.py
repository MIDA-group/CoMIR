import torch
import numpy as np

def mode2img(img, modslice):
    if isinstance(img, torch.Tensor):
        img = tensor2np(img)
    img = img[..., modslice]
    if img.shape[-1] == 1:
        return img[..., 0]
    else:
        return img

def tensor2np(tensor):
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0].cpu().detach().numpy()
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor[:, 0].cpu().detach().numpy()

    if tensor.ndim == 3:
        return tensor.permute(1, 2, 0).cpu().detach().numpy()
    elif tensor.ndim == 2:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.permute(0, 2, 3, 1).cpu().detach().numpy()

def img2uint8(img, a=None, b=None):
    if isinstance(img, torch.Tensor):
        img = tensor2np(img)
    if a is None and b is None:
        a, b = img.min(), img.max()
    return (255 * (img - a) / (b - a)).astype(np.uint8)

def normalize(a):
    if isinstance(a, torch.Tensor):
        a = tensor2np(a)
    a_min, a_max = np.percentile(a, [1, 99])
    new_a = np.clip((a - a_min) / (a_max - a_min), 0, 1)
    return new_a

def normalize_common(a, b):
    if isinstance(a, torch.Tensor):
        a = tensor2np(a)
    if isinstance(b, torch.Tensor):
        b = tensor2np(b)
    a_min, a_max = np.percentile(a, [1, 99])
    b_min, b_max = np.percentile(b, [1, 99])
    minimum = max(a_min, b_min)
    maximum = min(a_max, b_max)
    new_a = np.clip((a - minimum) / (maximum - minimum), 0, 1)
    new_b = np.clip((b - minimum) / (maximum - minimum), 0, 1)
    return new_a, new_b
