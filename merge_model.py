import sys
import os
from typing import Optional, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from calflops import calculate_flops
from timm.models.layers import trunc_normal_
from timm.models.resnet import resnet10t
import torch.nn.functional as F
import numpy as np


class SEBlock(nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        num_conv_branches: int = 1,
    ) -> None:
        """Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileOne(nn.Module):
    """MobileOne Model

    Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        init: bool = True,
        drop: float = 0.0,
        in_channels: int = 1,
        num_blocks_per_stage: List[int] = [2, 8, 10, 1],
        num_classes: int = 1000,
        width_multipliers: Optional[List[float]] = None,
        inference_mode: bool = False,
        use_se: bool = False,
        num_conv_branches: int = 1,
    ) -> None:
        """Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(
            in_channels=in_channels,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            inference_mode=self.inference_mode,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0
        )
        self.stage2 = self._make_stage(
            int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0
        )
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0,
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.flat = nn.Flatten()
        self.num_features = int(512 * width_multipliers[3])
        self.drop = drop
        self.dropout = nn.Dropout(p=drop)
        self.num_classes = num_classes
        self.classifier = (
            nn.utils.weight_norm(nn.Linear(self.num_features, num_classes))
            if num_classes > 0
            else nn.Identity()
        )
        # self.classifier = nn.utils.weight_norm(nn.Linear(self.embedding, num_classes))

        # self.stage_list = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4, self.gap, self.flat, self.linear])
        if init:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_stage(
        self, planes: int, num_blocks: int, num_se_blocks: int
    ) -> nn.Sequential:
        """Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError(
                    "Number of SE blocks cannot " "exceed number of layers."
                )
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            # Pointwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(
        self, x, target=None, mixup_hidden=False, mixup_alpha=None
    ) -> torch.Tensor:
        """Apply forward pass."""

        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        if self.drop > 0:
            x = self.dropout(x)

        x = self.classifier(x)
        return x


PARAMS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},
}


def mobileone(
    init: bool = True,
    drop: float = 0.0,
    in_channels: int = 1,
    num_classes: int = 1000,
    inference_mode: bool = False,
    variant: str = "s0",
) -> nn.Module:
    """Get MobileOne model.

    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model."""
    variant_params = PARAMS[variant]
    return MobileOne(
        init=init,
        drop=drop,
        in_channels=in_channels,
        num_classes=num_classes,
        inference_mode=inference_mode,
        **variant_params,
    )


class MainAssistAttention(nn.Module):
    def __init__(self, in_channels=64):
        super(MainAssistAttention, self).__init__()

        # Projection layers for attention
        self.q_proj = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k_proj = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v_proj = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, main_input, assist_inputs):
        # main_input shape: (1, 242, 300)
        # assist_inputs shape: (n_devices, 1, 242, 300)
        output = torch.zeros_like(main_input)
        # Project main input to Q
        # print("main_input.shape", main_input.shape)
        # print("assist_inputs.shape", assist_inputs[0].shape)
        q = self.q_proj(main_input)
        # print("q.shape", q.shape)
        output = main_input

        # Process each assist input
        for assist_input in assist_inputs:

            # Project assist input to K and V
            k = self.k_proj(assist_input)
            # print("k.shape", k.shape)
            v = self.v_proj(main_input)
            # print("v.shape", v.shape)

            # Compute attention sc and time scores
            # print("k.transpose(-2, -1).shape", k.transpose(-2, -1).shape)
            attn_time_scores = torch.matmul(q, k.transpose(-2, -1)) / (
                k.shape[-2] ** 0.5
            )
            # print("attn_time_scores.shape", attn_time_scores.shape)
            attn_time_probs = F.softmax(attn_time_scores, dim=-1)
            # print("attn_time_probs.shape", attn_time_probs.shape)

            # Compute attention scores
            attn_sc_scores = torch.matmul(q.transpose(-2, -1), k) / (k.shape[-1] ** 0.5)
            # print("attn_sc_scores.shape", attn_sc_scores.shape)
            attn_sc_probs = F.softmax(attn_sc_scores, dim=-1)
            # print("attn_sc_probs.shape", attn_sc_probs.shape)

            # Apply attention to values
            time_context = torch.matmul(attn_time_probs, v)
            # print("time_context.shape", time_context.shape)
            sc_context = torch.matmul(v, attn_sc_probs)
            # print("sc_context.shape", sc_context.shape)
            # Add to the output
            output = output + time_context + sc_context
            # print("main_input.shape", main_input.shape)

        return output


class RssiFuseV2(nn.Module):
    """
    RSSIFUSE fuse base on rssi weight
    """

    def __init__(self, channels=64):
        super(RssiFuseV2, self).__init__()

        # Projection layers for attention
        self.q_proj1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k_proj1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v_proj1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # # Projection layers for attention
        # self.q_proj2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        # self.k_proj2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        # self.v_proj2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(
        self, x1, x2, rssi1, rssi2, layer_mix=None, target=None, mixup_alpha=None
    ):
        rssi1_q = self.q_proj1(rssi1)
        rssi2_q = self.q_proj1(rssi2)

        csi1_k = self.k_proj1(x1)
        csi2_k = self.k_proj1(x2)

        csi1_v = self.v_proj1(x1)
        csi2_v = self.v_proj1(x2)

        rssi1_score = F.softmax(
            torch.matmul(rssi1_q, csi1_k.transpose(-2, -1)) / (csi1_k.shape[-2] ** 0.5),
            dim=-1,
        )

        rssi2_score = F.softmax(
            torch.matmul(rssi2_q, csi2_k.transpose(-2, -1)) / (csi2_k.shape[-2] ** 0.5),
            dim=-1,
        )

        csi1 = torch.matmul(rssi1_score, csi1_v)
        csi2 = torch.matmul(rssi2_score, csi2_v)

        fused_csi = csi1 + csi2
        return self.relu(self.bn(fused_csi))


class RssiFuseV1(nn.Module):
    """
    RSSIFUSE fuse base on rssi weight
    """

    def __init__(self, init, channels=64):
        super(RssiFuseV1, self).__init__()

        self.exp_hidden = channels * 4

        self.local_att = nn.Sequential(
            # pw
            nn.Conv1d(
                in_channels=channels,
                out_channels=self.exp_hidden,
                kernel_size=250,
                stride=250,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(self.exp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=self.exp_hidden,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.Softmax(dim=-1),
        )

        if init:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x1, x2, rssi1, rssi2, layer_mix=None, target=None, mixup_alpha=None
    ):

        # print("x1.shape", x1.shape)
        # print("x2.shape", x2.shape)
        # print("rssi1.shape", rssi1.shape)
        # print("rssi2.shape", rssi2.shape)
        concatenated = torch.cat(
            (rssi1, rssi2),
            dim=-1,
        )  # Concatenated along the channel dimension
        # print("concatenated.shape", concatenated.shape)
        rssi_local_weight = self.local_att(concatenated)
        # print("rssi_local_weight.shape", rssi_local_weight.shape)
        weil1, weil2 = torch.chunk(rssi_local_weight, 2, dim=-1)
        # print("weil1.shape", weil1.shape)
        # print("weil2.shape", weil2.shape)
        # output = self.relu(wei1.unsqueeze(-1) * x1 + wei2.unsqueeze(-1) * x2 + x1 + x2)
        # return output

        re_x1 = weil1.unsqueeze(-1) * x1
        re_x2 = weil2.unsqueeze(-1) * x2

        # fused_csi = x1 + x2
        fused_csi = torch.cat((re_x1, re_x2), dim=2)

        return re_x1, re_x2, fused_csi

        # if layer_mix is not None:
        #     fused_csi, y_a, y_b, lam = mixup_process(fused_csi, target, mixup_alpha)
        #     return fused_csi, y_a, y_b, lam
        # else:
        #     return fused_csi


def mixup_feature_process(out, lam, indices):
    out = out * lam + out[indices, :] * (1 - lam)
    return out


class EdgeClassifier(nn.Module):
    def __init__(self, init, in_channel=1, out_channel=64, num_classes=21):
        super(EdgeClassifier, self).__init__()

        expand_channels = out_channel * 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=expand_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=expand_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=expand_channels,
                out_channels=out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            # SEBlock(in_channels=out_channel),
        )

        self.conv_rssi = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=out_channel,
                out_channels=expand_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=expand_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=expand_channels,
                out_channels=out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(inplace=True),
            # SE1DBlock(in_channels=out_channel),
        )

        # self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.drop = drop
        # self.classifier = (
        #     nn.Linear(out_channel, num_classes, bias=True)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        if init:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, rssi):
        # x = self.conv(in_x) + in_x
        x = self.conv(x)
        # rssi_x = self.conv_rssi(in_rssi) + in_rssi
        rssi = self.conv_rssi(rssi)
        # x = self.gap(x)
        # x = x.view(x.size(0), -1)
        # if self.drop > 0:
        #     x = F.dropout(x, p=self.drop, training=self.training)
        # x = self.classifier(x)
        return x, rssi


# Define a convolutional layer for downsampling and fusion
class AttentionFuse(nn.Module):
    def __init__(self, init, in_channel=1, device_num=2):
        super(AttentionFuse, self).__init__()
        # Convolutional layers to downsample the inputs
        self.conv = nn.ModuleList(
            [
                EdgeClassifier(
                    init, in_channel=in_channel, out_channel=64, num_classes=0
                )
                for _ in range(device_num)
            ]
        )
        # Attention module to fuse the downsampled data
        self.attention_fuse = RssiFuseV1(init, channels=64)

    def forward(self, x, layer_mix=None, target=None, mixup_alpha=None):
        x1, rssi1 = self.conv[0](x["x1"], x["rssi1"])
        x2, rssi2 = self.conv[1](x["x2"], x["rssi2"])

        # print("x1.shape", x1.shape)
        # print("x2.shape", x2.shape)
        # print("rssi1.shape", rssi1.shape)
        # print("rssi2.shape", rssi2.shape)

        if layer_mix is not None:
            if mixup_alpha > 0.0:
                lam = (
                    torch.from_numpy(
                        np.array([np.random.beta(mixup_alpha, mixup_alpha)])
                    )
                    .float()
                    .cuda()
                )
            else:
                lam = 1.0

            batch_size = rssi1.size()[0]
            indices = torch.randperm(batch_size).cuda()

            y_a, y_b = target, target[indices]

            x1 = mixup_feature_process(x1, lam, indices)
            x2 = mixup_feature_process(x2, lam, indices)

            fused_csi = self.attention_fuse(x1, x2, rssi1, rssi2, layer_mix)
            # print("fused_x.shape", fused_x.shape)

            return fused_csi, y_a, y_b, lam

        else:
            re_x1, re_x2, fused_csi = self.attention_fuse(x1, x2, rssi1, rssi2)
            # print("fused_x.shape", fused_x.shape)

            return fused_csi

        # if layer_mix is not None:
        #     fused_csi, y_a, y_b, lam = self.attention_fuse(
        #         x1, x2, rssi1, rssi2, layer_mix, target, mixup_alpha
        #     )
        #     return fused_csi, y_a, y_b, lam
        # else:
        #     fused_csi = self.attention_fuse(x1, x2, rssi1, rssi2, layer_mix)
        #     # print("fused_x.shape", fused_x.shape)

        #     return fused_csi


class CatFuse(nn.Module):
    def __init__(self, device_num=2):
        super(CatFuse, self).__init__()
        # Convolutional layers to downsample the inputs
        self.conv = nn.ModuleList(
            [
                EdgeClassifier(in_channel=1, out_channel=64, num_classes=0)
                for _ in range(device_num)
            ]
        )
        self.conv[0].conv_rssi = nn.Identity()
        self.conv[1].conv_rssi = nn.Identity()

    def forward(self, x1, x2, layer_mix=None, target=None, mixup_alpha=None):

        x1 = self.conv[0].conv(x1)
        x2 = self.conv[1].conv(x2)

        # fused_csi = x1 + x2
        fused_csi = torch.cat((x1, x2), dim=2)

        return x1, x2, fused_csi

        # if layer_mix is not None:
        #     fused_csi, y_a, y_b, lam = mixup_process(fused_csi, target, mixup_alpha)
        #     return fused_csi, y_a, y_b, lam
        # else:
        #     return fused_csi


class SingleCNN(nn.Module):
    def __init__(self, num_classes=21) -> None:
        super().__init__()

        self.downsample = EdgeClassifier(in_channels=1, out_channels=64)

        self.backbone = mobileone(
            drop=0.0, in_channels=64, num_classes=21, variant="s4"
        )

        self.classifier = nn.Linear(self.backbone.embedding, num_classes, bias=True)

        # print(self.classifier.weight.shape)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x1.shape", x[0].shape)
        # print("x2.shape", x[1].shape)
        x = self.downsample.conv(x)
        x = self.backbone(x)
        # print(x.shape)
        x = self.classifier(x)
        return x


class ServerMerge(nn.Module):
    def __init__(self, num_classes=21) -> None:
        super().__init__()

        self.attention_fuse = AttentionFuse()

        self.backbone = mobileone(
            drop=0.0, in_channels=128, num_classes=21, variant="s4"
        )

        self.classifier = nn.utils.weight_norm(
            nn.Linear(self.backbone.embedding, num_classes, bias=True)
        )

        # print(self.classifier.weight.shape)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # print("x1.shape", x[0].shape)
        # print("x2.shape", x[1].shape)
        x = self.attention_fuse(x1, x2)
        x = self.backbone(x)
        # print(x.shape)
        x = self.classifier(x)
        return x


class TaskConsistencyLoss(nn.Module):
    def __init__(self, num_positions, num_classes, epsilon=1e-8):
        """
        Consistency loss for different task types

        Args:
            epsilon: Small value for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon
        # Project distributions to same space (optional)
        proj_dim = min(num_positions * 2, num_classes)
        self.pos_proj = nn.Linear(num_positions * 2, proj_dim)
        self.count_proj = nn.Linear(num_classes, proj_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_position_distribution(self, pos_logits):
        """
        Convert binary position logits to probability distribution
        Args:
            pos_logits: Shape (batch_size, num_positions)
        Returns:
            prob distribution: Shape (batch_size, num_positions * 2)
        """
        # Convert each binary position to a two-class distribution
        pos_probs = torch.sigmoid(pos_logits)
        # Stack probabilities for each position [p, 1-p]
        pos_dist = torch.stack([pos_probs, 1 - pos_probs], dim=-1)
        # Reshape to (batch_size, num_positions * 2)
        return pos_dist.reshape(pos_dist.size(0), -1)

    def _get_counting_distribution(self, count_logits):
        """
        Convert counting logits to probability distribution
        Args:
            count_logits: Shape (batch_size, num_classes)
        """
        return F.softmax(count_logits, dim=-1)

    def forward(self, pos_logits, count_logits, mask=None):
        """
        Calculate consistency between binary position and multi-class counting tasks

        Args:
            pos_logits: Binary position logits (batch_size, num_positions)
            count_logits: Counting logits (batch_size, num_count_classes)
            mask: Optional mask for valid positions
        """
        # Get probability distributions
        pos_dist = self._get_position_distribution(
            pos_logits
        )  # (batch_size, num_positions * 2)
        count_dist = self._get_counting_distribution(
            count_logits
        )  # (batch_size, num_count_classes)

        pos_proj_dist = F.softmax(self.pos_proj(pos_dist), dim=-1)
        count_proj_dist = F.softmax(self.count_proj(count_dist), dim=-1)

        # Calculate mean distribution
        m = 0.5 * (pos_proj_dist + count_proj_dist)

        # Calculate JS divergence
        kl1 = F.kl_div(
            (m + self.epsilon).log(), pos_proj_dist + self.epsilon, reduction="none"
        ).sum(-1)

        kl2 = F.kl_div(
            (m + self.epsilon).log(), count_proj_dist + self.epsilon, reduction="none"
        ).sum(-1)

        js_div = 0.5 * (kl1 + kl2)

        if mask is not None:
            js_div = js_div * mask

        return js_div.mean()


class AttentionMerge(nn.Module):
    def __init__(self, args, num_classes=21, in_channel=1) -> None:
        super().__init__()

        init = True
        if args.aug == "hp_dwt_sg":
            init = False

        self.attention_fuse = AttentionFuse(init, in_channel=in_channel)

        # self.backbone = mobileone(
        #     init, drop=0.0, in_channels=64, num_classes=0, variant="s4"
        # )

        self.backbone = resnet10t(in_chans=64, num_classes=num_classes)
        self.backbone.fc = nn.Identity()
        self.num_features = self.backbone.num_features

        self.classifier = nn.utils.weight_norm(
            nn.Linear(self.num_features, num_classes)
        )

        # self.knn_down = nn.Sequential(
        #     nn.Linear(self.backbone.num_features, self.backbone.num_features // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.backbone.num_features // 2, self.backbone.num_features // 4),
        #     nn.ReLU(),
        # )

        # self.knn_classifier = nn.utils.parametrizations.weight_norm(
        #     nn.Linear(self.backbone.num_features // 4, num_classes)
        # )

        if init:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x,
        # alpha=0.1,
        # beta=0.05,
    ) -> torch.Tensor:
        """
        Forward pass with task-specific losses and consistency

        Args:
            x1: Input head csi tensor
            x2: Input end csi tensor
            rssi1: Input head rssi tensor
            rssi2: Input end rssi tensor
        """
        feat = self.attention_fuse(x)
        feat = self.backbone(feat)
        # pos_logits = self.pos_head(x)
        out = self.classifier(feat)

        # knn_feat = self.knn_down(feat)

        # knn_x = self.knn_classifier(knn_feat)

        # return x, knn_x
        return out

        # # Position loss (binary cross entropy for each position)
        # pos_loss = self.pos_loss(
        #     pos_logits,
        #     pos_labels,
        # )

        # # Counting loss (cross entropy for multi-class)
        # count_loss = self.count_loss(
        #     count_logits,
        #     count_labels,
        # )

        # # Consistency loss
        # consistency_loss = self.consistency_loss(pos_logits, count_logits)

        # # Mutual information loss
        # mi_loss = -torch.mean(
        #     torch.sum(
        #         F.sigmoid(self.pos_proj(pos_logits))
        #         * F.softmax(self.count_proj(count_logits), dim=-1),
        #         dim=1,
        #     )
        # )

        # Combined loss
        # total_loss = (
        #     0.1 * pos_loss + 0.9 * count_loss
        # )  # + alpha * consistency_loss + beta * mi_loss

        # return {
        #     # "pos_logits": pos_logits,
        #     "count_logits": count_logits,
        #     # "loss": total_loss,
        #     # "pos_loss": pos_loss,
        #     "loss": count_loss,
        #     # "consistency_loss": consistency_loss,
        #     # "mi_loss": mi_loss,
        # }


if __name__ == "__main__":
    mobile = AttentionMerge(num_classes=21, rssi=False)
    flops, macs, params = calculate_flops(
        model=mobile,
        input_shape=(2, 1, 1, 300, 242),
        output_as_string=True,
        include_backPropagation=True,
        output_precision=3,
    )
    print("ConvNet4 FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
