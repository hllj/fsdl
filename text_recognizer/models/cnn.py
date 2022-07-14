from email.policy import default
from typing import Any, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28

class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x + identity)

class CNN(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)

        elf.conv1 = ConvBlock(input_dims[0], conv_dim)
        self.conv2 = ConvBlock(conv_dim, conv_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.max_pool = nn.MaxPool2d(2)

        conv_output_size = IMAGE_SIZE // 2
        fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--conv_dim', type=int, default=CONV_DIM)
        parser.add_argument('--fc_dim', type=int, default=FC_DIM)
        return parser