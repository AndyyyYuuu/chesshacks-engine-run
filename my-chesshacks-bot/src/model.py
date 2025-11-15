import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out
    

class Evaluator(nn.Module):
    def __init__(self, in_channels=18, channels=64, n_blocks=4):
        super(Evaluator, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_blocks)]
        )
        self.conv_out = nn.Conv2d(channels, 1, kernel_size=1)
        self.fc = nn.Linear(8 * 8, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.res_blocks(out)
        out = self.conv_out(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = Evaluator(in_channels=19, channels=32, n_blocks=8)
print(f"Total parameters: {count_parameters(model):,}")
