import torch
import torch.nn as nn

class MyModel(torch.nn.Module):
    def __init__(self):
        """
        Dummy model sample
        """
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
        return self.layers(x)

