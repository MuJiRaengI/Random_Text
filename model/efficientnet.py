import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.efficient = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        if in_channels != 3:
            out_feature = self.efficient.features[0][0].out_channels
            self.efficient.features[0][0] = nn.Conv2d(
                in_channels,
                out_feature,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        in_features = self.efficient.classifier[1].in_features

        self.efficient.classifier[1] = nn.Linear(in_features, out_channels)

    def forward(self, x):
        x = self.efficient(x)
        return F.sigmoid(x)


def main():
    import numpy as np

    model = EfficientNet(3, 1)
    img = torch.from_numpy(np.zeros((2, 3, 224, 224), dtype=np.float32))
    output = model(img)
    print()


if __name__ == "__main__":
    main()
