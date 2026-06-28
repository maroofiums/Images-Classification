import torch
import torch.nn as nn

from src.config import NUM_CLASSES

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.block(X)
    

class SimpleCNN(nn.Module):
    def __init__(self, nums_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,32),
            ConvBlock(32,32),
            nn.MaxPool2d(2),

            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),

            # 8 x 8
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, nums_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def build_model(num_classes=NUM_CLASSES):
    return SimpleCNN(nums_classes=num_classes)

if __name__ == "__main__":

    model = build_model()

    x = torch.randn(8, 3, 32, 32)

    output = model(x)

    print(model)

    print("Output Shape:", output.shape)