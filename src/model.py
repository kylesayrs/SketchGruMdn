from typing import Tuple, List

import torch
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily
)

from AutoEncoder import AutoEncoder


class SketchRNN(torch.nn.Module):
    def __init__(
        self,
        image_size: int = 50,
        num_components: int = 20,
        hidden_size: int = 256
    ):
        super().__init__()
        self.image_size = image_size
        self.num_components = num_components
        self.hidden_size = hidden_size

        self.image_encoder = AutoEncoder.decoder(
            image_size=image_size,
            latent_size=hidden_size
        )

        self.rnn = torch.nn.GRU(
            input_size=,
            hidden_size=hidden_size,
            num_layers=1
        )

        self.rnn_input_shape = (
            5 * num_components +  # mean_x, mean_y, std_x, std_y, correlation_xy (tril)
            1 * num_components +  # mixture_logits
            3 # pen state
        )

        # validate model shapes are correct
        self.validate()

    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        psuedo code

        """
        hidden_zero = self.image_encoder(images)
        input_zero = torch.zeros(images.shape[0], self.rnn_input_shape)

        output, hidden = self.rnn(input_zero, hidden_zero)

        pis = output[:, :self.num_components]  # M
        mus = output[:, ?: ? + self.num_components * 2].reshape(self.num_components, 2)  # 2 * m
        trils = output[:, ?: ? + self.num_components * 3].convert_to_trils(self.num_components, 2)
        pen_state = output[:, ?: ? + 3]

        mixture = Categorical(logits=pis)
        components = MultivariateNormal(mus, scale_tril=trils)
        mixture_model = MixtureSameFamily(mixture, components)

        delta_x, delta_y = mixture_model.sample(2)

        return torch.tensor([delta_x, delta_y] + pen_state)


    def validate(self):
        test_input = torch.rand((1, 1, self.image_size, self.image_size))
        self.forward(test_input)
