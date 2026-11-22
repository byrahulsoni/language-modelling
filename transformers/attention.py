import torch
from torch import nn
import math
from torch.nn import functional as F
from typing import Annotated


BoolTensor = Annotated[torch.Tensor, torch.bool]
FloatTensor = Annotated[torch.Tensor, torch.float32]


def scaled_dot_product_attn(
        Q: FloatTensor, # [..., seq_length, d_k]
        K: FloatTensor, # [..., seq_length, d_k]
        V: FloatTensor, # [..., seq_length, d_v]
        mask: BoolTensor = None,
        dropout: nn.Linear = None
        ) -> FloatTensor: # [batch, seq_length, seq_length]
    """
    Compute masked (and scaled) self-attention for a given sequence
    Q, K, V can have following shapes:
    [batch, seq_length, d_model] for single hand attention, or
    [batch, num_head, seq_length, d_k] for multi-head hand attention
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) # output: [..., seq_len, seq_len]
    scores = scores / math.sqrt(d_k) # output: [..., seq_len, seq_len]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    scores = F.softmax(scores, dim=-1) # output: [..., seq_len, seq_len]
    if dropout is not None:
        scores = dropout(scores) # must pass instance of nn.Linear for dropout
    output = scores @ V # output: [..., seq_len, d_v]
    return output, scores


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int = None,
                 dropout: int = 0.0, bias: bool = False
                 ) -> None:
        
        super().__init__()

        d_v = d_v or d_k
        self.W_q = nn.Linear(in_features=d_model, out_features=d_k)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_k)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_v)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input: FloatTensor, mask: BoolTensor | list = None) -> tuple:
        # torch operates linear layer as X * W_transpose
        # where W is stored as: [out_features, in_features], so [d_attn, d_model] 
        # input: [batch, seq_len, d_model] where d_model is also emb dimension
        q = self.W_q(input) # output: [batch, seq_len, d_attn]
        k = self.W_k(input) # output: [batch, seq_len, d_attn]
        v = self.W_v(input) # output: [batch, seq_len, d_attn]
        output, attn = scaled_dot_product_attn(q, k, v, mask, self.dropout)
        return output, attn # [batch, seq_len, d_v], [batch, seq_len, seq_len]


class NaiveMultiHeadSelfAttention(nn.Module):
    """
    Using SingleHeadAttention to create multi head attention
    Advantage: Easy to follow and builds on single head attention
    Disadvantage: SLOW.. because we are not utilising parallel matmul by stacking weight matrices
    """
    def __init__(self, d_model: int, h: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        d_k = d_model // h

        heads = [
            SingleHeadAttention(
                d_model=d_model,
                d_k=d_k,
                dropout=dropout,
                bias=bias
            )
            for _ in range(h)
            ]
        self.heads = nn.ModuleList(heads)
        self.W_o = nn.Linear(in_features=int(d_k * h), out_features=d_model, bias=bias)

    def forward(self, input: FloatTensor) -> FloatTensor:
        # each item in list of shape: [batch, seq_len, d_k]
        output = [head(input)[0] for head in self.heads]
        output = torch.cat(output, dim=-1) # output: [batch, seq_len, d_k*h]
        final_output = self.W_o(output)
        return final_output # [batch, seq_len, d_model]


class MultiHeadSelfAttention(nn.Module):
    """
    Efficient implementation by using stacking matrices of each head into a single matrix
    Results in faster matmul
    """
    def __init__(self, d_model: int, num_heads: int, d_k: int = None,
                 d_v: int = None, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.h = num_heads
        self.d_k = d_k or d_model // num_heads
        self.d_v = d_v or d_model // num_heads

        self.W_q = nn.Linear(in_features=d_model, out_features=num_heads * d_k, bias=bias) # h heads of (d_model x d_k)
        self.W_k = nn.Linear(in_features=d_model, out_features=num_heads * self.d_k, bias=bias) # []
        self.W_v = nn.Linear(in_features=d_model, out_features=num_heads * self.d_v, bias=bias) # []
        self.W_o = nn.Linear(in_features=num_heads * self.d_v, out_features=d_model, bias=bias) # []
        self.dropout = nn.Dropout(dropout)


    def split_heads(self, tensor: FloatTensor, head_size: int) -> FloatTensor:
        """
        Split input tensor into h heads
        head_dim is either d_k or d_v
        Input: [batch_size, seq_len, h*d_k]
        Output: [batch_size, num_head, seq_len, d_k]
        PS: If d_k and d_v were same, we would just use self.d_k and wouldn't require this arg
        """
        batch_size, seq_len, _ = tensor.shape
        output = tensor.view(batch_size, seq_len, self.h, head_size)
        output = output.transpose(-2, -3) # output: [batch, num_head, seq_len, head_dim]
        return output


    def merge_heads(self, tensor: FloatTensor, head_size: int) -> FloatTensor:
        tensor = tensor.transpose(1, 2) # output: [batch, seq_len, num_head, d_v]
        tensor = tensor.view(batch, seq_len, -1) # output: [batch, seq_len, num_head * d_v]
        return tensor

    def forward(self, input: FloatTensor, mask: BoolTensor = None) -> FloatTensor:
        Q = self._split_heads(self.W_q(input)) # [batch, num_head, seq_len, d_k]
        K = self._split_heads(self.W_k(input)) # [batch, num_head, seq_len, d_k]
        V = self._split_heads(self.W_v(input)) # [batch, num_head, seq_len, d_v]
        output, scores = scaled_dot_product_attn(Q, K, V, mask=mask, dropout=self.dropout)
        # output: [batch, num_head, seq_len, d_v]
        # scroes: [batch, num_head, seq_len, seq_len]
        output = self.merge_heads(output) # [batch, seq_len, num_head * d_v]
        output = self.W_o(output) # output: [batch_size, seq_len, d_model]
        return output, scores # [batch, seq_len, d_model] and [batch, seq_len, seq_len]



# if __name__ == "__main__":
#     batch = 4
#     seq_len = 10
#     d_model = 512 # also equal to embedding dim
#     d_k=128
#     d_v=64
#     attention = SingleHeadAttention(d_model, d_k, d_v)
#     input = torch.rand((batch, seq_len, d_model))
#     output, attn_scores = attention(input)
#     print(input.shape)
#     print(attn_scores.shape)
#     print(attn_scores[0])
#     print(torch.sum(attn_scores[0], dim=1))
#     print(output.shape)


def causal_mask(n, device=None):
    """[1, 1, n, n] lower-triangular bool mask: position i may attend to j <= i."""
    return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device)).view(1, 1, n, n)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, seq_len, d_model, num_heads = 2, 10, 512, 8
    x = torch.randn(B, seq_len, d_model)

    # --- Single head (rectangular weights: 512 -> 64) ---
    single = SingleHeadAttention(d_model, d_k=64)
    out, attn = single(x)
    print("input tensor     :", tuple(x.shape))     # (2, 10, 512)
    print("output tensor    :", tuple(out.shape))     # (2, 10, 64)
    print("attention scores :", tuple(attn.shape))    # (2, 10, 10)
    print("W_q weight       :", tuple(single.W_q.weight.shape), "(PyTorch stores [out, in])")

    # --- Multi-head self-attention with a causal mask ---
    mhsa = MultiHeadSelfAttention(d_model, num_heads)
    out, attn = mhsa(x, mask=causal_mask(seq_len))
    print("MHSA out        :", tuple(out.shape))     # (2, 10, 512)
    print("MHSA attn       :", tuple(attn.shape))    # (2, 8, 10, 10)
    print("causal upper-triangle all zero:", bool(attn.triu(1).abs().max() == 0))
    exit()

    # --- Show fused MHSA == literal per-head version (copy weights across) ---
    def copy_into_naive(mhsa, naive):
        with torch.no_grad():
            for i, head in enumerate(naive.heads):
                for name, hd in (("W_q", mhsa.d_k), ("W_k", mhsa.d_k), ("W_v", mhsa.d_v)):
                    rows = slice(i * hd, (i + 1) * hd)   # head i = a slice of the stacked weight
                    getattr(head, name).weight.copy_(getattr(mhsa, name).weight[rows])
                    getattr(head, name).bias.copy_(getattr(mhsa, name).bias[rows])
            naive.W_o.load_state_dict(mhsa.W_o.state_dict())

    naive = NaiveMultiHeadSelfAttention(d_model, h)
    copy_into_naive(mhsa, naive)
    print("fused == per-head naive:", torch.allclose(naive(x, causal_mask(n)), out, atol=1e-5))

    # --- d_k != d_v (and h*d_k != d_model): e.g. 8 heads, d_k=48, d_v=96 ---
    mhsa2 = MultiHeadSelfAttention(d_model, h, d_k=48, d_v=96)
    out2, attn2 = mhsa2(x, mask=causal_mask(n))
    print("W_q/W_k/W_v/W_o :", [tuple(getattr(mhsa2, m).weight.shape) for m in ("W_q", "W_k", "W_v", "W_o")])
    print("d_k!=d_v out    :", tuple(out2.shape), " attn:", tuple(attn2.shape))
    naive2 = NaiveMultiHeadSelfAttention(d_model, h, d_k=48, d_v=96)
    copy_into_naive(mhsa2, naive2)
    print("d_k!=d_v fused == naive:", torch.allclose(naive2(x, causal_mask(n)), out2, atol=1e-5))

    # --- Sanity check against PyTorch's built-in kernel ---
    q = torch.randn(B, h, n, 64); k = torch.randn(B, h, n, 64); v = torch.randn(B, h, n, 64)
    ours, _ = scaled_dot_product_attention(q, k, v, causal_mask(n))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print("matches F.scaled_dot_product_attention:", torch.allclose(ours, ref, atol=1e-5))