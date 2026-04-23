# =============================================================================
# partial_conv.py  –  Partial Convolution Layer (Liu et al., 2018)
# =============================================================================
# Paper: "Image Inpainting for Irregular Holes Using Partial Convolutions"
#         https://arxiv.org/abs/1804.07723
#
# Core idea:
#   Standard convolution treats masked (missing) pixels the same as valid ones,
#   leading to artifacts near hole boundaries. Partial convolution:
#     1. Applies the conv kernel ONLY over valid (known) pixels.
#     2. Renormalises the output by the ratio of kernel_area / num_valid_pixels,
#        so the feature magnitude is independent of how many pixels were masked.
#     3. Auto-updates the binary mask: any output position that had ≥1 valid
#        input pixel becomes valid in the next layer's mask.
#
# This makes it ideal for inpainting because:
#   - Features near hole boundaries are not "contaminated" by zeroed hole pixels.
#   - The mask shrinks naturally through the encoder, expanding back in decoder.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """
    Partial Convolution 2D layer.

    Forward signature:
        output, updated_mask = layer(input_feat, mask)

    Args:
        in_channels  : number of input feature channels
        out_channels : number of output feature channels
        kernel_size  : convolution kernel size (int or tuple)
        stride       : convolution stride
        padding      : convolution padding
        dilation     : convolution dilation
        bias         : whether to add a learnable bias

    Tensor conventions (shapes documented with B=batch, C=channels, H=height, W=width):
        input_feat   : (B, in_channels,  H,   W)   float32 — features (0 in holes)
        mask         : (B, 1,            H,   W)   float32 — 1=valid, 0=hole
        output       : (B, out_channels, H',  W')  float32 — renormalised features
        updated_mask : (B, 1,            H',  W')  float32 — grown mask
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 bias: bool = True):
        super().__init__()

        self.feature_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # Mask convolution: fixed all-ones weights, no bias, no grad.
        # It counts how many valid pixels fall under each kernel position.
        self.mask_conv = nn.Conv2d(
            1, 1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        nn.init.constant_(self.mask_conv.weight, 1.0)

        # Mask convolution weights are fixed — never updated by optimiser.
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        # Pre-compute total number of kernel elements for renormalisation.
        # kernel_size can be int or tuple.
        ks = kernel_size if isinstance(kernel_size, (tuple, list)) \
             else (kernel_size, kernel_size)
        self.kernel_area = float(ks[0] * ks[1])   # = sum(all-ones mask kernel)

        # Initialise feature conv weights (He init works well with ReLU)
        nn.init.kaiming_normal_(self.feature_conv.weight,
                                mode='fan_in', nonlinearity='relu')
        if self.feature_conv.bias is not None:
            nn.init.zeros_(self.feature_conv.bias)

    def forward(self, input_feat: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            input_feat : (B, in_channels, H, W)  — zero-filled in hole regions
            mask       : (B, 1, H, W)             — 1=valid, 0=hole

        Returns:
            out_feat     : (B, out_channels, H', W')  — renormalised output
            updated_mask : (B, 1, H', W')             — 1 wherever ≥1 valid pixel
        """
        # ── Step 1: Mask-gated feature convolution ────────────────────────────
        # Zero out hole pixels before convolving so they don't contribute.
        # mask is broadcast across in_channels.
        # (B, in_channels, H, W) * (B, 1, H, W) → (B, in_channels, H, W)
        masked_input = input_feat * mask

        # Standard conv over masked features
        # (B, in_channels, H, W) → (B, out_channels, H', W')
        raw_out = self.feature_conv(masked_input)

        # ── Step 2: Count valid pixels under each kernel position ─────────────
        # (B, 1, H, W) → (B, 1, H', W')
        with torch.no_grad():
            mask_count = self.mask_conv(mask)   # sum of valid pixels in kernel

        # Clamp to avoid division by zero (happens for fully-masked positions)
        mask_count_safe = mask_count.clamp(min=1e-8)

        # ── Step 3: Renormalise output ────────────────────────────────────────
        # Scale each output position by (kernel_area / num_valid_pixels).
        # Positions fully inside the hole have mask_count→0, output stays 0.
        #   renorm_factor : (B, 1, H', W')
        renorm_factor = self.kernel_area / mask_count_safe

        # Only renormalise where at least one valid pixel existed
        valid_positions = (mask_count > 0).float()   # (B, 1, H', W')
        out_feat = raw_out * renorm_factor * valid_positions

        # ── Step 4: Update the mask ───────────────────────────────────────────
        # Any output position that had ≥1 valid input pixel becomes valid.
        # (B, 1, H', W')  in {0.0, 1.0}
        updated_mask = valid_positions

        return out_feat, updated_mask


# ─────────────────────────────────────────────────────────────────────────────
# PartialConvBlock — Partial Conv + BN + activation, reusable in encoder
# ─────────────────────────────────────────────────────────────────────────────

class PartialConvBlock(nn.Module):
    """
    PartialConv2d → BatchNorm2d → ReLU block.

    Used in the U-Net encoder where mask-awareness is critical.
    Skip connections pass BOTH the feature map AND the updated mask
    to the corresponding decoder block.

    Forward:
        out_feat, updated_mask = block(input_feat, mask)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_bn: bool = True,
                 activation: bool = True):
        super().__init__()

        self.pconv = PartialConv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn  = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)        if activation else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x    : (B, in_channels,  H,  W)
            mask : (B, 1,            H,  W)
        Returns:
            out  : (B, out_channels, H', W')
            mask : (B, 1,            H', W')
        """
        out, mask = self.pconv(x, mask)   # partial conv + mask update
        out = self.bn(out)
        out = self.act(out)
        return out, mask


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B = 2
    x    = torch.randn(B, 2, 64, 64)            # 2-channel input (corrupted mask + hole mask)
    mask = torch.ones(B, 1, 64, 64)             # all valid
    mask[:, :, 20:40, 20:40] = 0.0              # punch a 20x20 hole

    pconv = PartialConvBlock(in_channels=2, out_channels=64, stride=2)
    out, updated_mask = pconv(x, mask)

    # (B, 2, 64, 64) → stride-2 pconv → (B, 64, 32, 32)
    print(f"Input  : {x.shape}")
    print(f"Mask   : {mask.shape}")
    print(f"Output : {out.shape}")            # expect (2, 64, 32, 32)
    print(f"UpdMask: {updated_mask.shape}")   # expect (2, 1,  32, 32)
    print("✅ PartialConv2d smoke test passed")
