from torch import nn
from mmdet.registry import MODELS
import torch
from torch import Tensor
from mmdet.registry import DATASETS
import numpy as np
from mmdet.models.backbones.self_prompt_tuning import RepCatcher
class AuxiliaryBranch(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims, int(dims // 16)),
            nn.Linear(int(dims // 16), dims),
            nn.SiLU(),
        )
        # self.mask_ratio = nn.Parameter(torch.tensor(0.2))

    def fourier_transform(self, feats):
        B, L, C = feats.shape

        # 找到最接近的2的幂次方
        next_power_of_two = 1
        while next_power_of_two < L:
            next_power_of_two *= 2

        # 填充到 next_power_of_two
        padded_feats = torch.nn.functional.pad(feats, (0, 0, 0, next_power_of_two - L))  # 填充到最近的2的幂次方

        # 傅里叶变换
        fft_feats = torch.fft.fft(padded_feats, dim=1)  # 沿序列维度 L（填充后）进行FFT

        # 生成高频掩码（保留中心区域的高频成分）
        # mask = torch.zeros(next_power_of_two, dtype=torch.float32, device=feats.device)
        # center = next_power_of_two // 2
        # mask_width = int(next_power_of_two * self.mask_ratio) // 2  # 高频区域宽度
        # mask_start = max(0, center - mask_width)
        # mask_end = min(next_power_of_two, center + mask_width)
        # mask[mask_start:mask_end] = 1.0  # 中心区域置1

        # 应用掩码（保留高频成分）
        # masked_fft = fft_feats * mask[None, :, None]  # 维度广播：[B, L_padded, C]

        # 傅里叶逆变换
        ifft_feats = torch.fft.ifft(fft_feats, dim=1).real  # 取实部

        # 截取原长度
        ifft_feats = ifft_feats[:, :L, :]

        return ifft_feats

    def forward(self, x):
        x = self.mlp(x)
        out = self.fourier_transform(x)
        return out + x


class CausalBranch(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU()
        )

    def forward(self, x):
        return self.mlp(x)


@MODELS.register_module()
class Cauvis(nn.Module):
    def __init__(
            self,
            num_layers: int,
            embed_dims: int,
            patch_size: int,
            img_size: int,
            prompt_init=None,
            query_dims: int = 256,
            token_length: int = 100,
            use_softmax: bool = True,
            link_token_to_query: bool = True,
            scale_init: float = 0.001,
            zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.token_embedding = int(img_size // patch_size) * int(img_size // patch_size)

        self.prompt = nn.Parameter(torch.zeros([self.token_length, self.embed_dims]))

        self.mlp_prompt = nn.Linear(self.embed_dims, self.embed_dims)
        self.to_out = nn.Linear(self.embed_dims, self.embed_dims)
        self.alpha = nn.Parameter(torch.tensor(self.scale_init))
        self.beta = nn.Parameter(torch.tensor(self.scale_init))
        self.delta_scale = nn.Parameter(torch.tensor(1.0))

        self.prompt_branch = CausalBranch(self.embed_dims)
        self.aux_branch = AuxiliaryBranch(self.embed_dims)

    def cross_attention(self, x, prompt):
        attn = torch.einsum('bnc,mc->bnm', x, prompt)
        attn = attn * (self.embed_dims ** -0.5)
        attn = attn.softmax(-1)
        score = torch.einsum('bnm,mc->bnc', attn, self.mlp_prompt(prompt))
        return self.to_out(score)

    def forward(
            self, feats: Tensor,
            layer: int,
            batch_first=False,
            has_cls_token=True
    ) -> Tensor:
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=1)

        res_prompt = self.cross_attention(feats, self.prompt)
        main = self.prompt_branch(res_prompt) * self.alpha
        aux = self.aux_branch(res_prompt) * self.beta
        feats = feats * self.delta_scale + main + aux

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=1)
        return feats  # bnc
