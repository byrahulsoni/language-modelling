import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import math


class PositionalEncodingNumpy:
    """Numpy implementation of Positional Encoding"""
    def __init__(self, d_model:int, max_len:int=2048, constant_base:float=10000.0) -> None:
        # denominator = exp((i / d) * log(10000.0)) = exp (i * log(N)/d_model)
        # multiplier = denominator^(-1) = exp(i * -log(N)/d_model)
        # denominator does not change for sine and cosine so:
        # multiplier = row vector of size [d_model // 2, 1]
        constant = -np.log(constant_base) / d_model
        multiplier = np.exp(np.arange(0, d_model, 2) * constant)

        self.pe = np.zeros((max_len, d_model)) # position-encoding
        positions = np.arange(max_len)[:, np.newaxis] # (max_len) -> [max_len, 1] colm vector
        # encoding_matrix[:, 0::2] = [max_len, d_model // 2]
        # outer product of colm vector and row vector:
        # (max_len, 1) x (1, d_model // 2) -> resulting in slice [max_len, d_model//2]
        self.pe[:, 0::2] = np.sin(positions * multiplier)
        self.pe[:, 1::2] = np.cos(positions * multiplier)

    def compute(self, seq_length: int) -> NDArray[np.float32]:
        return self.pe[:seq_length]


class PositionalEncodingPyTorch(nn.Module):
    """PyTorch implementation of Positional Encoding"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len:int=5000, constant_base:float=10000.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # denominator = exp((i / d) * log(10000.0)) = exp (i * log(N)/d_model)
        # multiplier = denominator^(-1) = exp(i * -log(N)/d_model)
        # denominator does not change for sine and cosine so:
        # multiplier = row vector of size [d_model // 2, 1]
        constant = -math.log(constant_base) / d_model
        multiplier = torch.exp(torch.arange(0, d_model, 2) * constant)

        positions = torch.arange(max_len).unsqueeze(-1) # (max_len, 1)
        pe = torch.zeros((1, max_len, d_model)) # (batch, max_len, embedding_dim)
        # encoding_matrix[:, 0::2] = [max_len, d_model // 2]
        # outer product of colm vector and row vector:
        # (max_len, 1) x (1, d_model // 2) -> resulting in slice [max_len, d_model//2]
        pe[0, :, 0::2] = torch.sin(positions * multiplier)
        pe[0, :, 1::2] = torch.cos(positions * multiplier)
        self.register_buffer('pe', pe) # ignored from parameters, still part of state_dict()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor of shape: ``[batch_size, seq_len, embedding_dim]``
        """
        # x -> (b, seq_len, emb_dim)
        # pe -> (b, max_seq_len, emb_dim)
        # pe[:,x.size(1), :] -> (b, seq_len, emb_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def get_pe_matrix(encoder: PositionalEncodingNumpy | PositionalEncodingPyTorch, seq_len: int) -> NDArray:
    """Return the encoding as a numpy array of shape (seq_len, d_model) for either implementation."""
    pe = encoder.pe

    if isinstance(pe, Tensor):
        pe = pe.detach().cpu().numpy()
    # Torch buffer carries a leading batch dim of 1; numpy version does not.
    if pe.ndim == 3:
        pe = pe[0]
    if seq_len > pe.shape[0]:
        raise ValueError(
            f"seq_len={seq_len} exceeds encoder max_len={pe.shape[0]}; "
            f"construct {type(encoder).__name__} with a larger max_len"
        )
    return pe[:seq_len]


def plot_pe(
    encoder_cls: type = PositionalEncodingNumpy,
    seq_lens: tuple[int, ...] = (128, 256, 512),
    d_models: tuple[int, ...] = (128, 256, 512),
    line_dims: tuple[int, ...] = (0, 16, 32, 64),
) -> None:
    max_len = max(seq_lens)
    encoders = {d: encoder_cls(d_model=d, max_len=max_len) for d in d_models}

    # Analyise one param change at a time (seq_len versus d_model)
    n_rows, n_cols = len(seq_lens), len(d_models)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
    )

    im = None
    for r, seq_len in enumerate(seq_lens):
        for c, d_model in enumerate(d_models):
            pe = get_pe_matrix(encoders[d_model], seq_len)
            ax = axes[r, c]
            im = ax.imshow(pe, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_title(f'seq_len={seq_len}, d_model={d_model}')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Position')

    fig.suptitle(f'Sinusoidal Positional Encoding ({encoder_cls.__name__})')
    fig.tight_layout()
    # Shared colorbar: all panels use the same [-1, 1] range.
    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.show()

    # Line plot: same grid, one subplot per (seq_len, d_model), showing selected dimensions over position.
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False, sharey=True
    )
    for r, seq_len in enumerate(seq_lens):
        for c, d_model in enumerate(d_models):
            pe = get_pe_matrix(encoders[d_model], seq_len)
            ax = axes[r, c]
            # Skip dims that don't exist for smaller d_model rather than raising IndexError.
            valid_dims = [d for d in line_dims if 0 <= d < d_model]
            for d in valid_dims:
                ax.plot(pe[:, d], label=f'Dim {d}')
            ax.set_title(f'seq_len={seq_len}, d_model={d_model}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Value')
            if valid_dims:
                ax.legend(fontsize='small')

    fig.suptitle(f'PE values for selected dimensions ({encoder_cls.__name__})')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # plot_pe(PositionalEncodingNumpy)
    plot_pe(PositionalEncodingPyTorch)
