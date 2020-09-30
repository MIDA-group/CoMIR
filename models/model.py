# Python Standard Library imports
import logging

# Other libraries
from abc import ABC

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(
        self,
        in_channels=3,
        nb_classes=1,
        activation_func=None,
        activation_kwargs=None,
        epistemic=None,
        ensemble_size=1,
        aleatoric=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.nb_classes = nb_classes
        self.activation_func = activation_func
        self.activation_kwargs = activation_kwargs if activation_kwargs else {}
        self.epistemic = epistemic
        self.ensemble_size = ensemble_size
        self.aleatoric = aleatoric

    def __str__(self):
        return type(self).__qualname__

    def get_param_count(self):
        param_count_nogrd = 0
        param_count_grd = 0
        for param in self.parameters():
            if param.requires_grad:
                param_count_grd += param.size().numel()
            else:
                param_count_nogrd += param.size().numel()
        return param_count_grd, param_count_nogrd

    def summary(self, half=False, printf=print):
        """ Logs some information about the neural network.
        Args:
            printf: The printing function to use.
        """
        layers_count = len(list(self.modules()))
        print(f"Model {self} has {layers_count} layers.")
        param_grd, param_nogrd = self.get_param_count()
        param_total = param_grd + param_nogrd
        print(f"-> Total number of parameters: {param_total:n}")
        print(f"-> Trainable parameters:       {param_grd:n}")
        print(f"-> Non-trainable parameters:   {param_nogrd:n}")
        approx_size = param_total * (2.0 if half else 4.0) * 10e-7
        print(f"Uncompressed size of the weights: {approx_size:.1f}MB")

    def save(self, filename):
        """Saves the model"""
        torch.save({"state_dict": self.state_dict()}, filename)

    def half(self):
        """ Transforms all the weights of the model in half precision.

        Note: this function fixes an issue on BatchNorm being half precision.
            See: https://discuss.pytorch.org/t/training-with-half-precision/11815/2
        """
        super(BaseModel, self).half()
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    def initialize_kernels(
        self, initializer, conv=False, linear=False, batchnorm=False, **kwargs
    ):
        """ Initializes the chosen set of kernels of the model.

        Args:
            initializer: a function that will apply the weight initialization.
            conv: Will initialize the kernels of the convolutions.
            linear: Will initialize the kernels of the linear layers.
            batchnorm: Will initialize the kernels of the batch norm layers.
            **kwargs: Extra arguments to pass to the initializer function.
        """
        for layer in self.modules():
            if (
                (linear and isinstance(layer, nn.Linear))
                or (conv and isinstance(layer, nn.Conv2d))
                or (batchnorm and isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)))
            ):
                initializer(layer.weight, **kwargs)

    def initialize_biases(
        self, initializer, conv=False, linear=False, batchnorm=False, **kwargs
    ):
        """Initializes the chosen set of biases of the model.

        Args:
            initializer: A function that will apply the weight initialization.
            conv: Will initialize the biases of the convolutions.
            linear: Will initialize the biases of the linear layers.
            batchnorm: Will initialize the biases of the batch norm layers.
            **kwargs: Extra arguments to pass to the initializer function.
        """
        for layer in self.modules():
            if (
                (linear and isinstance(layer, nn.Linear))
                or (conv and isinstance(layer, nn.Conv2d))
                or (batchnorm and isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)))
            ):
                initializer(layer.bias, **kwargs)
