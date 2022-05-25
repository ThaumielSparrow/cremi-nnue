import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture, implemented to follow schema in relevant paper: https://arxiv.org/abs/1505.04597
    
    Batch normalization added between convolve layers for reduction in covariance
    """
    def __init__(self, in_chan=3, out_chan=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # UNet encoder blocks
        for feature in features:
            self.downs.append(DoubleConv(in_chan, feature))
            in_chan = feature

        # UNet decoder blocks
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Misc UNet utils
        self.plateau = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_chan, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.plateau(x)
        skip_connections.reverse()

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Barrier against error when inputs are not factors of 16 
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


if __name__ == "__main__":
    x = torch.randn((3, 1, 150, 150))
    model = UNet(in_chan=1, out_chan=1)
    preds = model(x)
    print(preds.shape, x.shape)
    assert preds.shape == x.shape