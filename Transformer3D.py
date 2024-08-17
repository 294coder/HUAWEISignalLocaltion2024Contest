import math
from functools import partial
from typing import Literal, Union

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def cube_root(n):
    return round(math.pow(n, (1 / 3)))


class MLP_(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        # added batchnorm (remove it ?)
        x = self.bn(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # added batchnorm (remove it ?)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x):
        B, N, C = x.shape
        # (batch, patch_cube, hidden_size) -> (batch, hidden_size, D, H, W)
        # assuming D = H = W, i.e. cube root of the patch is an integer number!
        n = cube_root(N)
        x = x.transpose(1, 2).view(B, C, n, n, n)
        x = self.dwconv(x)
        # added batchnorm (remove it ?)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class _MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        # embedding dimesion of each attention head
        self.attention_head_dim = embed_dim // num_heads

        # The same input is used to generate the query, key, and value,
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, attention_head_size)
        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (batch_size, num_patches, hidden_size)
        B, N, C = x.shape

        # (batch_size, num_head, sequence_length, embed_dim)
        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            n = cube_root(N)
            # (batch_size, sequence_length, embed_dim) -> (batch_size, embed_dim, patch_D, patch_H, patch_W)
            x_ = x.permute(0, 2, 1).reshape(B, C, n, n, n)
            # (batch_size, embed_dim, patch_D, patch_H, patch_W) -> (batch_size, embed_dim, patch_D/sr_ratio, patch_H/sr_ratio, patch_W/sr_ratio)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # (batch_size, embed_dim, patch_D/sr_ratio, patch_H/sr_ratio, patch_W/sr_ratio) -> (batch_size, sequence_length, embed_dim)
            # normalizing the layer
            x_ = self.sr_norm(x_)
            # (batch_size, num_patches, hidden_size)
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            # (2, batch_size, num_heads, num_sequence, attention_head_dim)
        else:
            # (batch_size, num_patches, hidden_size)
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            # (2, batch_size, num_heads, num_sequence, attention_head_dim)

        k, v = kv[0], kv[1]

        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.num_heads)
        attnention_prob = attention_score.softmax(dim=-1)
        attnention_prob = self.attn_dropout(attnention_prob)
        out = (attnention_prob @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        mlp_ratio: at which rate increasse the projection dim of the embedded patch in the _MLP component
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=0.0)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, transf=False):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.transf = transf
        if transf:
            self.t_blk = TransformerBlock(embed_dim=in_planes, sr_ratio=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        if self.transf:
            b, c, d, h, w = out.shape
            out = rearrange(out, "b c d h w -> b (d h w) c")
            out = self.t_blk(out)
            out = rearrange(out, "b (d h w) c -> b c d h w", d=d, h=h, w=w)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, transf=False):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.transf = transf
        if transf:
            self.t_blk = TransformerBlock(embed_dim=in_planes, sr_ratio=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        if self.transf:
            b, c, d, h, w = out.shape
            out = rearrange(out, "b c d h w -> b (d h w) c")
            out = self.t_blk(out)
            out = rearrange(out, "b (d h w) c -> b c d h w", d=d, h=h, w=w)

        return out


class Transformer3d(nn.Module):
    """
        Args:
            blockType (Union[BasicBlock, Bottleneck]): Type of block used in the model.
            nLayers (list[int]): List of integers representing the number of layers in each block.
            blockInputChannels (list[int]): List of integers representing the number of input channels for each block.
            nC (int, optional): Number of input channels. Defaults to 4.
            InputConvTSize (int, optional): Size of the input convolutional kernel. Defaults to 7.
            InputConvTStride (int, optional): Stride of the input convolutional kernel. Defaults to 1.
            NotHasMaxPool (bool, optional): Flag indicating whether to use max pooling. Defaults to False.
            residualTyp (str, optional): Type of residual connection. Defaults to "B".
            Factor (float, optional): Factor to scale the number of input channels. Defaults to 1.
            outputC (int, optional): Number of output channels. Defaults to 2.
        Attributes:
            blockInputChannels (int): Number of input channels for the first block.
            notHasMaxPool (bool): Flag indicating whether max pooling is used.
            conv1 (nn.Conv3d): 3D convolutional layer for the input.
            bn1 (nn.BatchNorm3d): Batch normalization layer.
            relu (nn.ReLU): ReLU activation function.
            maxpool (nn.MaxPool3d): Max pooling layer.
            layer1 (nn.Sequential): First layer of blocks.
            layer2 (nn.Sequential): Second layer of blocks.
            layer3 (nn.Sequential): Third layer of blocks.
            layer4 (nn.Sequential): Fourth layer of blocks.
            avgpool (nn.AdaptiveAvgPool3d): Adaptive average pooling layer.
            fc (nn.Linear): Fully connected layer for the output.
        Methods:
            _downsample_basic_block(x, planes, stride): Downsamples the input tensor.
            _make_layer(block, planes, blocks, shortcut_type, stride=1): Creates a layer of blocks.
            forward(x): Forward pass of the model.
    """
    def __init__(
        self,
        blockType: "Union[BasicBlock, Bottleneck]",
        nLayers: list[int],
        blockInputChannels: list[int],
        nC: int = 4,
        InputConvTSize: int = 7,
        InputConvTStride: int = 1,
        NotHasMaxPool: bool = False,
        residualTyp: str = "B",
        Factor: float = 1.,
        outputC: int = 2,
    ):
        super().__init__()

        blockInputChannels = [int(x * Factor) for x in blockInputChannels]

        self.blockInputChannels = blockInputChannels[0]
        self.notHasMaxPool = NotHasMaxPool

        self.conv1 = nn.Conv3d(
            nC,
            self.blockInputChannels,
            kernel_size=(InputConvTSize, 7, 7),
            stride=(InputConvTStride, 2, 2),
            padding=(InputConvTSize // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.blockInputChannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            blockType, 
            blockInputChannels[0],
            nLayers[0], 
            residualTyp
        )
        self.layer2 = self._make_layer(
            blockType, 
            blockInputChannels[1], 
            nLayers[1],
            residualTyp, 
            stride=2
        )
        self.layer3 = self._make_layer(
            blockType, 
            blockInputChannels[2], 
            nLayers[2], 
            residualTyp, 
            stride=2
        )
        self.layer4 = self._make_layer(
            blockType, 
            blockInputChannels[3], 
            nLayers[3], 
            residualTyp,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(blockInputChannels[3] * blockType.expansion, outputC)

    def _downsample_basic_block(self, x, planes, stride):
        xx = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zPads = torch.zeros(
            xx.size(0), 
            planes - xx.size(1),
            xx.size(2), 
            xx.size(3), 
            xx.size(4)
        )
        if isinstance(xx.data, torch.cuda.FloatTensor):
            zPads = zPads.cuda()

        xx = torch.cat([xx.data, zPads], dim=1)

        return xx

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.blockInputChannels != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.blockInputChannels, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.blockInputChannels,
                planes=planes,
                stride=stride,
                downsample=downsample,
                transf=False,
            )
        )
        self.blockInputChannels = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.blockInputChannels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.notHasMaxPool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def used_model():
    return Transformer3d(Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512])
