# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding,
# )
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 192
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 512
    multiple_of: int = 16  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim,
                            args.n_heads * self.head_dim,
                            bias=True,)
        # ColumnParallelLinear(
        #     args.dim,
        #     args.n_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wk = nn.Linear(args.dim,
                            args.n_kv_heads * self.head_dim,
                            bias=False,)
        # ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wv = nn.Linear(args.dim,
                            args.n_kv_heads * self.head_dim,
                            bias=False,)
        # ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wo = nn.Linear(args.n_heads * self.head_dim,
                            args.dim,
                            bias=False,)
        # RowParallelLinear(
        #     args.n_heads * self.head_dim,
        #     args.dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )

        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        keys = xk
        values = xv

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        # # repeat k/v heads if n_kv_heads < n_heads
        # keys = repeat_kv(
        #     keys, self.n_rep
        # )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # values = repeat_kv(
        #     values, self.n_rep
        # )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        
        # math
        # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # if mask is not None:
        #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # torch
        output = F.scaled_dot_product_attention(xq, keys, values, mask, dropout_p=0.0, scale=1 / math.sqrt(self.head_dim))
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        self.w2 = nn.Linear(hidden_dim, dim)
        # RowParallelLinear(
        #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        self.w3 = nn.Linear(dim, hidden_dim)
        # ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs=None):
        super().__init__()
        if params is None:
            params = ModelArgs()
        
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # self.tok_embeddings = VocabParallelEmbedding(
        #     params.vocab_size, 
        #     params.dim, 
        #     init_method=lambda x: x
        # )
        self.to_latent = nn.Linear(150, params.dim)
        # ColumnParallelLinear(1, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        # ColumnParallelLinear(
        #     params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        # )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )
        
        self.to_pos = nn.Linear(params.dim, 2, bias=True)

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int=0):
        # tokens = tokens.permute(0, 1, -1, 2)
        tokens = tokens.flatten(2).permute(0, 2, 1)
        
        _bsz, seqlen, _ = tokens.shape
        # h = tokens
        # h = self.tok_embeddings(tokens)
        h = self.to_latent(tokens)
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        # if seqlen > 1:
        #     mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

        #     mask = torch.triu(mask, diagonal=1)

        #     # When performing key-value caching, we compute the attention scores
        #     # only for the new sequence. Thus, the matrix of scores is of size
        #     # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        #     # j > cache_len + i, since row i corresponds to token cache_len + i.
        #     mask = torch.hstack(
        #         [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
        #     ).type_as(tokens)

        # breakpoint()
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # h = self.norm(h)
        # h = self.output(h).float()
        output = self.to_pos(h.mean(dim=1))
        return output
    
    
        
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        array_count = 4
        angle_dim = 128
        range_dim = 150
        slice_num = 571
        
        self.flatten = nn.Flatten(1, 3)
        self.linear1 = nn.Linear(array_count * angle_dim * range_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.linear4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.linear5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.linear6 = nn.Linear(16, 2)
        self.bn6 = nn.BatchNorm1d(2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.bn1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.linear3(x)
        x = self.relu(x)
        x = self.bn3(x)

        x = self.linear4(x)
        x = self.relu(x)
        x = self.bn4(x)

        x = self.linear5(x)
        x = self.relu(x)
        x = self.bn5(x)

        x = self.linear6(x)
        x = self.relu(x)
        x = self.bn6(x)
        return x
    
if __name__ == '__main__':
    import os
    # from fairscale.nn.model_parallel.initialize import (
    #                                                         get_model_parallel_rank,
    #                                                         initialize_model_parallel,
    #                                                         model_parallel_is_initialized,
    #                                                     )
    # from torch.distributed import is_nccl_available, is_initialized, init_process_group
    
    # model_parallel_size = 1
    
    # if not is_initialized() and is_nccl_available():
    #     init_process_group(backend="nccl", init_method='env://', world_size=model_parallel_size)
    # if not model_parallel_is_initialized():
    #     if model_parallel_size is None:
    #         model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    #     initialize_model_parallel(model_parallel_size)
        
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
        
    model = Transformer(params=ModelArgs()).cuda()
    inp_seq = torch.randn((1, 150, 128, 4)).cuda()
    
    print(model(inp_seq, 0).shape)
    
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
    
    print(
        flop_count_table(
            FlopCountAnalysis(model, (inp_seq, 0))
        )
    )
    
    
    
    # model = MLP()
    # print(parameter_count_table(model))
    