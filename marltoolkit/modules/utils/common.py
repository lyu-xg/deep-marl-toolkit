from typing import Any, Tuple

import torch
import torch.nn as nn


class MLPBase(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'relu',
        use_orthogonal: bool = False,
        use_feature_normalization: bool = False,
    ) -> None:
        super(MLPBase, self).__init__()

        use_relu = 1 if activation == 'relu' else 0
        active_func = [nn.ReLU(), nn.Tanh()][use_relu]
        if use_orthogonal:
            init_method = nn.init.orthogonal_
        else:
            init_method = nn.init.xavier_uniform_

        gain = nn.init.calculate_gain(['tanh', 'relu'][use_relu])
        self.use_feature_normalization = use_feature_normalization
        if use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        def init_weight(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                init_method(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), active_func,
                                 nn.LayerNorm(hidden_dim))
        self.fc2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), active_func,
                                 nn.LayerNorm(hidden_dim))
        self.apply(init_weight)

    def forward(self, inputs: torch.Tensor):
        if self.use_feature_normalization:
            output = self.feature_norm(inputs)

        output = self.fc1(output)
        output = self.fc2(output)

        return output


class Flatten(nn.Module):

    def forward(self, inputs: torch.Tensor):
        return inputs.view(inputs.size(0), -1)


class CNNBase(nn.Module):

    def __init__(
        self,
        obs_shape: Tuple,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = 'relu',
        use_orthogonal: bool = False,
    ) -> None:
        super(CNNBase, self).__init__()

        use_relu = 1 if activation == 'relu' else 0
        active_func = [nn.ReLU(), nn.Tanh()][use_relu]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_relu])

        (in_channel, width, height) = obs_shape

        def init_weight(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                init_method(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=hidden_dim // 2,
                kernel_size=kernel_size,
                stride=stride,
            ),
            active_func,
            Flatten(),
            nn.Linear(
                hidden_dim // 2 * (width - kernel_size + stride) *
                (height - kernel_size + stride),
                hidden_dim,
            ),
            active_func,
            nn.Linear(hidden_dim, hidden_dim),
            active_func,
        )
        self.apply(init_weight)

    def forward(self, inputs: torch.Tensor):
        inputs = inputs / 255.0
        output = self.cnn(inputs)
        return output


class RNNLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        rnn_hidden_dim: int,
        rnn_layers: int,
        use_orthogonal: bool = True,
    ) -> None:
        super(RNNLayer, self).__init__()
        self.rnn_layers = rnn_layers
        self.use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(input_dim, rnn_hidden_dim, num_layers=rnn_layers)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self.use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.layer_norm = nn.LayerNorm(rnn_hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[Any, torch.Tensor]:
        if inputs.size(0) == hidden_state.size(0):
            inputs = inputs.unsqueeze(0)
            hidden_state = (hidden_state * masks.repeat(
                1, self.rnn_layers).unsqueeze(-1).transpose(0, 1).contiguous())
            output, hidden_state = self.rnn(inputs, hidden_state)
            output = output.squeeze(0)
            hidden_state = hidden_state.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hidden_state.size(0)
            T = int(inputs.size(0) / N)
            # unflatten
            inputs = inputs.view(T, N, inputs.size(1))
            # Same deal with masks
            masks = masks.view(T, N)
            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(
                dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hidden_state = hidden_state.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hidden_state * masks[start_idx].view(1, -1, 1).repeat(
                    self.rnn_layers, 1, 1)).contiguous()
                rnn_scores, hidden_state = self.rnn(inputs[start_idx:end_idx],
                                                    temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            inputs = torch.cat(outputs, dim=0)

            # flatten
            inputs = inputs.reshape(T * N, -1)
            hidden_state = hidden_state.transpose(0, 1)

        output = self.layer_norm(inputs)
        return output, hidden_state
