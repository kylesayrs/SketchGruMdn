from typing import Tuple, List

import torch


def conv_down(in_filters: int, out_filters: int, bn: bool = True) -> List[torch.nn.Module]:
    block = [
        torch.nn.BatchNorm2d(in_filters) if bn else None,
        torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.2)
    ]
    block = [layer for layer in block if layer is not None]
    
    return block


def conv_up(in_filters: int, out_filters: int, bn: bool = True) -> List[torch.nn.Module]:
    block = [
        torch.nn.BatchNorm2d(in_filters) if bn else None,
        torch.nn.ConvTranspose2d(in_filters, out_filters, 3, 2, 1),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.2)
    ]
    block = [layer for layer in block if layer is not None]
    
    return block


def linear_block(in_features: int, out_features: int, bn: bool = True) -> List[torch.nn.Module]:
    block = [
        torch.nn.BatchNorm2d(in_features) if bn else None,
        torch.nn.Linear(in_features, out_features, 3, 2, 1),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.2)
    ]
    block = [layer for layer in block if layer is not None]
    
    return block


def conv_up_final(in_filters: int, out_filters: int) -> List[torch.nn.Module]:
    return [
        torch.nn.BatchNorm2d(in_filters),
        torch.nn.ConvTranspose2d(in_filters, out_filters, 3, 2, 1),
        torch.nn.Sigmoid()
    ]


class AutoEncoder(torch.nn.Module):
    def __init__(self, image_size: int = 50, latent_size: int = 256):
        super().__init__()
        self.image_size = image_size

        # define encoder
        self.encoder = torch.nn.Sequential(
            *conv_down(1, 32, bn=False),
            *conv_down(32, 64),
            *conv_down(64, 64),
            *conv_down(64, 64),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, latent_size),
        )

        # define decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (64, 4, 4)),
            *conv_up(64, 64),
            *conv_up(64, 64),
            *conv_up(64, 32),
            *conv_up(32, 1),
            torch.nn.Flatten(),
            torch.nn.Linear(2401, image_size * image_size),
            torch.nn.Unflatten(1, (1, image_size, image_size)),
            torch.nn.Sigmoid(),
        )

        # validate model shapes are correct
        self.validate()

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        z = self.encoder(x)
        y = self.decoder(z)

        return y, z


    def validate(self):
        test_input = torch.rand((1, 1, self.image_size, self.image_size))
        self.forward(test_input)
