import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool2d(nn.Sequential):
    """Blur Pooling Layer (MaxPool2d replacement)
    See: https://richzhang.github.io/antialiased-cnns/
    Paper: https://arxiv.org/abs/1904.11486
    """

    __constants__ = ["in_features"]
    _blur_kernel = torch.tensor(
        [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]]
    )

    def __init__(self, in_features):
        """
        Args:
            in_features (int): The number of channels in the input
        """
        super().__init__()
        self.in_features = in_features

        self.add_module("maxpool", nn.MaxPool2d(2, stride=1))
        blurpool = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False,
            groups=in_features,
        )
        blurpool.weight = torch.nn.Parameter(
            self._blur_kernel.repeat(in_features, 1, 1, 1), requires_grad=False
        )
        self.add_module("blurpool", blurpool)

    def forward(self, x):
        return super(BlurPool2d, self).forward(x)

    def extra_repr(self):
        return "in_features={}".format(self.in_features)
