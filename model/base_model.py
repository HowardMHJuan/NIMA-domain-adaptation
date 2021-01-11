import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Base class for models."""
    @abstractmethod
    def forward(self, *inputs):
        return NotImplemented

    def __str__(self):
        """For printing the model and the number of trainable parameters.

        Returns:
            (str) -- the model and the number of trainable parameters
        """
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        n_params = sum(
            [torch.prod(torch.tensor(p.size())) for p in trainable_params])

        separate_line_str = ('----------------------------------------'
                             '------------------------------\n')

        return '{0}{1}\n{0}Trainable parameters: {2}\n{0}'.format(
            separate_line_str, super().__str__(), n_params)
