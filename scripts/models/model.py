"""Model Class"""

import torch
from torch import nn

class MyModel(torch.nn.Module):
    """
    Dummy model sample
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(3,3,1),
                        nn.SiLU(),
                        nn.Conv2d(3,3,1),
                        nn.SiLU(),
                        nn.Conv2d(3,1,1),
                        nn.SiLU()
                    )

    def forward(self, x):
        """
        The forward method of the model
        """
        return self.layers(x)
