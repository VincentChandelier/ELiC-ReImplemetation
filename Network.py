"""
A simple test algorithm to rewrite the network
"""
import math
import time

import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import trunc_normal_
from ELICUtilis.layers import (
    AttentionBlock,
    conv3x3,
    CheckboardMaskedConv2d,
)

from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.ops import ste_round
from compressai.models.utils import conv, deconv, update_registered_buffers

from thop import profile
from ptflops import get_model_complexity_info

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out


class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)


class TestModel(CompressionModel):
    def __init__(
        self,
        N=192,
        M=320,
        num_slices=5,
        groups=[0, 16, 16, 32, 64, 192],
        **kwargs,
    ):
        """ELIC 2022; uneven channel groups with checkerboard context.

        Context model from [He2022], with minor simplifications.
        Based on modified attention model architecture from [Cheng2020].

        .. note::

            This implementation contains some differences compared to the
            original [He2022] paper. For instance, the implemented context
            model only uses the first and the most recently decoded channel
            groups to predict the current channel group. In contrast, the
            original paper uses all previously decoded channel groups.

        [He2022]: `"ELIC: Efficient Learned Image Compression with
        Unevenly Grouped Space-Channel Contextual Adaptive Coding"
        <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
        Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

        [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
        Mixture Likelihoods and Attention Modules"
        <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
        Masaru Takeuchi, and Jiro Katto, CVPR 2020.

        Args:
             N (int): Number of main network channels
             M (int): Number of latent space channels
             num_slices (int): Number of slices/groups
             groups (list[int]): Number of channels in each channel group
        """
        super().__init__(entropy_bottleneck_channels=N)

        assert len(groups) == num_slices + 1
        assert sum(groups) == M
        assert groups[0] == 0

        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices
        self.groups = groups

        self.g_a = nn.Sequential(
            conv(3, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, M),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2),
            nn.ReLU(inplace=True),
            conv3x3(N * 3 // 2, 2 * M),
        )

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(
                    # Input: first group, and most recently decoded group.
                    self.groups[1] + (i > 1) * self.groups[i],
                    224,
                    stride=1,
                    kernel_size=5,
                ),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1] * 2, stride=1, kernel_size=5),
            )
            for i in range(1, num_slices)
        )  ## from https://github.com/tensorflow/compression/blob/master/models/ms2020.py

        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
                self.groups[i], self.groups[i] * 2, kernel_size=5, padding=2, stride=1
            )
            for i in range(1, num_slices + 1)
        )  ## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(
                    # Input: spatial context, channel context, and hyper params.
                    self.groups[i] * 2 + (i > 1) * self.groups[i] * 2 + M * 2,
                    M * 2,
                ),
                nn.ReLU(inplace=True),
                conv1x1(M * 2, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i] * 2),
            )
            for i in range(1, num_slices + 1)
        )  ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数

        self.quantizer = Quantizer()

        self.gaussian_conditional = GaussianConditional(None)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mode_quant="ste"):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        if mode_quant == "ste":
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = ste_round(z - z_offset) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # Extract anchor and non-anchors for each channel group:
        anchor = torch.zeros_like(y)
        non_anchor = torch.zeros_like(y)
        self._copy(anchor, y, "anchor")
        self._copy(non_anchor, y, "non_anchor")
        anchor_split = torch.split(anchor, self.groups[1:], dim=1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], dim=1)

        y_slices = torch.split(y, self.groups[1:], dim=1)

        y_likelihood = []
        y_hat_slices = []
        y_hat_slices_for_gs = []

        for slice_index, y_slice in enumerate(y_slices):
            support = self._calculate_support(
                slice_index, y_hat_slices, latent_means, latent_scales
            )

            y_hat_i, y_hat_for_gs_i, y_likelihood_i = self._checkerboard_forward(
                [y_slice, anchor_split[slice_index], non_anchor_split[slice_index]],
                slice_index,
                support,
                mode_quant=mode_quant,
            )

            y_hat_slices.append(y_hat_i)
            y_hat_slices_for_gs.append(y_hat_for_gs_i)
            y_likelihood.append(y_likelihood_i)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        y_slices = torch.split(y, self.groups[1:], dim=1)

        y_strings = []
        y_hat_slices = []
        params_start = time.time()

        for slice_index, y_slice in enumerate(y_slices):
            support = self._calculate_support(
                slice_index, y_hat_slices, latent_means, latent_scales
            )

            y_hat_i, y_strings_i = self._checkerboard_codec(
                [y_slice.clone(), y_slice.clone()],
                slice_index,
                support,
                y_shape=y.shape[-2:],
                mode="compress",
            )

            y_hat_slices.append(y_hat_i)
            y_strings.append(y_strings_i)

        params_time = time.time() - params_start

        strings = [y_strings, z_strings]

        return {
            "strings": strings,
            "shape": z.size()[-2:],
            "time": {
                "y_enc": y_enc,
                "z_enc": z_enc,
                "z_dec": z_dec,
                "params": params_time,
            },
        }

    def decompress(self, strings, shape, **kwargs):
        assert isinstance(strings, list) and len(strings) == 2
        [y_strings, z_strings] = strings

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        y_hat_slices = []

        for slice_index in range(len(self.groups) - 1):
            support = self._calculate_support(
                slice_index, y_hat_slices, latent_means, latent_scales
            )

            y_hat_i, _ = self._checkerboard_codec(
                y_strings[slice_index],
                slice_index,
                support,
                y_shape=(shape[0] * 4, shape[1] * 4),
                mode="decompress",
            )

            y_hat_slices.append(y_hat_i)

        y_hat = torch.cat(y_hat_slices, dim=1)

        y_dec_start = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        y_dec = time.time() - y_dec_start

        return {"x_hat": x_hat, "time": {"y_dec": y_dec}}

    def inference(self, x, mode_quant="ste"):
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start

        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        if mode_quant == "ste":
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = ste_round(z - z_offset) + z_offset

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        # Extract anchor and non-anchors for each channel group:
        anchor = torch.zeros_like(y)
        non_anchor = torch.zeros_like(y)
        self._copy(anchor, y, "anchor")
        self._copy(non_anchor, y, "non_anchor")
        anchor_split = torch.split(anchor, self.groups[1:], dim=1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], dim=1)

        y_slices = torch.split(y, self.groups[1:], dim=1)

        y_likelihood = []
        y_hat_slices = []
        params_start = time.time()

        for slice_index, y_slice in enumerate(y_slices):
            support = self._calculate_support(
                slice_index, y_hat_slices, latent_means, latent_scales
            )

            y_hat_i, _, y_likelihood_i = self._checkerboard_forward(
                [y_slice, anchor_split[slice_index], non_anchor_split[slice_index]],
                slice_index,
                support,
                mode_quant=mode_quant,
            )

            y_hat_slices.append(y_hat_i)
            y_likelihood.append(y_likelihood_i)

        params_time = time.time() - params_start
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat)
        y_dec = time.time() - y_dec_start

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "time": {
                "y_enc": y_enc,
                "y_dec": y_dec,
                "z_enc": z_enc,
                "z_dec": z_dec,
                "params": params_time,
            },
        }

    def _apply_quantizer(self, y, means, mode_quant):
        if mode_quant == "noise":
            quantized = self.quantizer.quantize(y, "noise")
            quantized_for_g_s = self.quantizer.quantize(y, "ste")
        elif mode_quant == "ste":
            quantized = self.quantizer.quantize(y - means, "ste") + means
            quantized_for_g_s = self.quantizer.quantize(y - means, "ste") + means
        else:
            raise ValueError(f"Unknown quantization mode: {mode_quant}")
        return quantized, quantized_for_g_s

    def _calculate_support(
        self, slice_index, y_hat_slices, latent_means, latent_scales
    ):
        if slice_index == 0:
            return torch.concat([latent_means, latent_scales], dim=1)

        cc_means, cc_scales = self.cc_transforms[slice_index - 1](
            y_hat_slices[0]
            if slice_index == 1
            else torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
        ).chunk(2, 1)

        return torch.concat([cc_means, cc_scales, latent_means, latent_scales], dim=1)

    def _checkerboard_forward(self, y_input, slice_index, support, mode_quant):
        y, y_anchor, y_non_anchor = y_input
        # NOTE: y == y_anchor + y_non_anchor

        B, C, H, W = y.shape
        means = torch.zeros_like(y)
        scales = torch.zeros_like(y)

        y_anchor_hat, y_anchor_hat_for_gs = self._checkerboard_forward_step(
            y_anchor,
            slice_index,
            support,
            means,
            scales,
            ctx_params=torch.zeros((B, C * 2, H, W), device=support.device),
            mode_quant=mode_quant,
            mode="anchor",
        )

        y_non_anchor_hat, y_non_anchor_hat_for_gs = self._checkerboard_forward_step(
            y_non_anchor,
            slice_index,
            support,
            means,
            scales,
            ctx_params=self.context_prediction[slice_index](y_anchor_hat),
            mode_quant=mode_quant,
            mode="non_anchor",
        )

        y_hat = y_anchor_hat + y_non_anchor_hat
        y_hat_for_gs = y_anchor_hat_for_gs + y_non_anchor_hat_for_gs

        # Entropy estimation
        _, y_likelihood = self.gaussian_conditional(y, scales, means=means)

        return y_hat, y_hat_for_gs, y_likelihood

    def _checkerboard_forward_step(
        self, y, slice_index, support, means, scales, ctx_params, mode_quant, mode
    ):
        means_new, scales_new = self.ParamAggregation[slice_index](
            torch.concat([ctx_params, support], dim=1)
        ).chunk(2, 1)

        self._copy(means, means_new, mode)
        self._copy(scales, scales_new, mode)

        y_hat, y_hat_for_gs = self._apply_quantizer(y, means_new, mode_quant)

        self._keep_only(y_hat, mode)
        self._keep_only(y_hat_for_gs, mode)

        return y_hat, y_hat_for_gs

    def _checkerboard_codec(self, y_input, slice_index, support, y_shape, mode):
        y_anchor_input, y_non_anchor_input = y_input

        # NOTE: y.shape == (B, C, H, W)
        B, *_ = support.shape
        C = self.groups[slice_index + 1]
        H, W = y_shape

        anchor_strings, y_anchor_hat = self._checkerboard_codec_step(
            y_anchor_input,
            slice_index,
            support,
            ctx_params=torch.zeros((B, C * 2, H, W), device=support.device),
            mode_codec=mode,
            mode_step="anchor",
        )

        non_anchor_strings, y_non_anchor_hat = self._checkerboard_codec_step(
            y_non_anchor_input,
            slice_index,
            support,
            ctx_params=self.context_prediction[slice_index](y_anchor_hat),
            mode_codec=mode,
            mode_step="non_anchor",
        )

        y_hat = y_anchor_hat + y_non_anchor_hat
        y_strings = [anchor_strings, non_anchor_strings]

        return y_hat, y_strings

    def _checkerboard_codec_step(
        self, y_input, slice_index, support, ctx_params, mode_codec, mode_step
    ):
        means_new, scales_new = self.ParamAggregation[slice_index](
            torch.concat([ctx_params, support], dim=1)
        ).chunk(2, 1)

        device = means_new.device
        decode_shape = means_new.shape
        B, C, H, W = decode_shape
        encode_shape = (B, C, H, W // 2)

        means = torch.zeros(encode_shape, device=device)
        scales = torch.zeros(encode_shape, device=device)
        self._unembed(means, means_new, mode_step)
        self._unembed(scales, scales_new, mode_step)

        indexes = self.gaussian_conditional.build_indexes(scales)

        if mode_codec == "compress":
            y = y_input
            y_encode = torch.zeros(encode_shape, device=device)
            self._unembed(y_encode, y, mode_step)
            strings = self.gaussian_conditional.compress(y_encode, indexes, means=means)

        elif mode_codec == "decompress":
            strings = y_input

        quantized = self.gaussian_conditional.decompress(strings, indexes, means=means)
        y_decode = torch.zeros(decode_shape, device=device)
        self._embed(y_decode, quantized, mode_step)

        return strings, y_decode

    def _copy(self, dest, src, mode):
        """Copy pixels in the current mode (i.e. anchor / non-anchor)."""
        if mode == "anchor":
            dest[:, :, 0::2, 0::2] = src[:, :, 0::2, 0::2]
            dest[:, :, 1::2, 1::2] = src[:, :, 1::2, 1::2]
        elif mode == "non_anchor":
            dest[:, :, 0::2, 1::2] = src[:, :, 0::2, 1::2]
            dest[:, :, 1::2, 0::2] = src[:, :, 1::2, 0::2]

    def _unembed(self, dest, src, mode):
        """Compactly extract pixels for the given mode.

        src                     dest

        ■ □ ■ □                 ■ ■           □ □
        □ ■ □ ■       --->      ■ ■           □ □
        ■ □ ■ □                 ■ ■           □ □
                                anchor        non-anchor
        """
        if mode == "anchor":
            dest[:, :, 0::2, :] = src[:, :, 0::2, 0::2]
            dest[:, :, 1::2, :] = src[:, :, 1::2, 1::2]
        elif mode == "non_anchor":
            dest[:, :, 0::2, :] = src[:, :, 0::2, 1::2]
            dest[:, :, 1::2, :] = src[:, :, 1::2, 0::2]

    def _embed(self, dest, src, mode):
        """Insert pixels for the given mode.

        src                                   dest

        ■ ■           □ □                     ■ □ ■ □
        ■ ■           □ □           --->      □ ■ □ ■
        ■ ■           □ □                     ■ □ ■ □
        anchor        non-anchor
        """
        if mode == "anchor":
            dest[:, :, 0::2, 0::2] = src[:, :, 0::2, :]
            dest[:, :, 1::2, 1::2] = src[:, :, 1::2, :]
        elif mode == "non_anchor":
            dest[:, :, 0::2, 1::2] = src[:, :, 0::2, :]
            dest[:, :, 1::2, 0::2] = src[:, :, 1::2, :]

    def _keep_only(self, y, mode):
        """Keep only pixels in the current mode, and zero out the rest."""
        if mode == "anchor":
            # Zero the non-anchors:
            y[:, :, 0::2, 1::2] = 0
            y[:, :, 1::2, 0::2] = 0
        elif mode == "non_anchor":
            # Zero the anchors:
            y[:, :, 0::2, 0::2] = 0
            y[:, :, 1::2, 1::2] = 0


if __name__ == "__main__":
    # model = convTransformer(H=384, W=384, lenslet_num=8, viewsize=6, C=128, depth=4, heads=4, dim_head=96, mlp_dim=96, dropout=0.1, emb_dropout=0.)
    # model = convTransformer(H=192, W=192, channels=3, patchsize=2, dim=64, depth=2, heads=4,
    #                         dim_head=64, mlp_dim=64, dropout=0.1,
    #                         emb_dropout=0.)

    model = TestModel(N=192, M=320, num_slices=5)
    # model = JointAutoregressiveHierarchicalPriors(192, 192)
    # model = Cheng2020Attention(128)
    input = torch.Tensor(1, 3, 256, 256)
    # from torchvision import models
    # model = models.resnet18()
    # print(model)
    out = model(input)
    print(out["x_hat"].shape)
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)