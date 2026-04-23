# =============================================================================
# inpainting_model.py  –  Partial Convolution U-Net for Road Mask Inpainting
# =============================================================================
# Architecture summary:
#
#   INPUT  : (B, 2, 512, 512)   ← [corrupted_mask, hole_mask] concatenated
#
#   ENCODER (Partial Conv — mask-aware):
#     enc1 : PConvBlock(2→64,   stride=1) → (B, 64,  512, 512)
#     pool1: stride-2 PConvBlock           → (B, 64,  256, 256)
#     enc2 : PConvBlock(64→128, stride=1) → (B, 128, 256, 256)
#     pool2: stride-2 PConvBlock           → (B, 128, 128, 128)
#     enc3 : PConvBlock(128→256,stride=1) → (B, 256, 128, 128)
#     pool3: stride-2 PConvBlock           → (B, 256,  64,  64)
#     enc4 : PConvBlock(256→512,stride=1) → (B, 512,  64,  64)
#
#   BOTTLENECK:
#     pool4: stride-2 PConvBlock           → (B, 512,  32,  32)
#     btn  : PConvBlock(512→512)           → (B, 512,  32,  32)
#
#   DECODER (standard convolutions + skip connections):
#     dec4 : upsample→(B,512,64,64)   + skip enc4 → Conv(1024→256)
#     dec3 : upsample→(B,256,128,128) + skip enc3 → Conv(512→128)
#     dec2 : upsample→(B,128,256,256) + skip enc2 → Conv(256→64)
#     dec1 : upsample→(B,64,512,512)  + skip enc1 → Conv(128→32)
#
#   OUTPUT: Conv(32→1) + Sigmoid → (B, 1, 512, 512)
#
# Channel progression note:
#   Decoder receives concatenated [upsampled, skip] so input channels double.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from partial_conv import PartialConvBlock


# ─────────────────────────────────────────────────────────────────────────────
# Decoder building block (standard Conv — no partial conv in decoder)
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsample × 2 → Concatenate skip → Conv3×3 → BN → ReLU.

    The decoder uses standard (non-partial) convolutions because by the
    time we reach the decoder, enough context has been gathered to fill
    the hole regions in feature space.  Using partial conv in the decoder
    adds complexity without meaningful benefit for mask inpainting.

    Args:
        in_channels  : channels of the upsampled feature map
        skip_channels: channels of the encoder skip connection
        out_channels : output channels after conv

    Forward:
        x    : (B, in_channels,  H,   W)
        skip : (B, skip_channels, H*2, W*2)
    Returns:
        out  : (B, out_channels, H*2, W*2)
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample to match skip connection spatial size
        # (B, in_channels, H, W) → (B, in_channels, H*2, W*2)
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)

        # Concatenate along channel dim
        # (B, in_channels+skip_channels, H*2, W*2)
        x = torch.cat([x, skip], dim=1)

        # (B, out_channels, H*2, W*2)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Partial Conv U-Net
# ─────────────────────────────────────────────────────────────────────────────

class PartialConvUNet(nn.Module):
    """
    U-Net with Partial Convolution encoder for road mask inpainting.

    INPUT  : (B, 2,   H, W)  — torch.cat([corrupted_mask, hole_mask], dim=1)
    OUTPUT : (B, 1,   H, W)  — completed road mask probabilities in [0, 1]

    Masks are propagated through the encoder and used at each partial conv
    layer. The decoder does NOT use the mask — standard convolutions only.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 64):
        """
        Args:
            in_channels   : number of input channels (2 for our task)
            base_channels : width multiplier; doubles at each encoder level
        """
        super().__init__()
        c = base_channels   # 64

        # ── Encoder (Partial Conv blocks) ─────────────────────────────────────
        # Each level: one partial conv block at stride=1, then one at stride=2
        # to halve spatial resolution (avoids MaxPool — keeps mask update clean).

        # Level 1: (B, 2, H, W) → (B, 64, H, W) → pool → (B, 64, H/2, W/2)
        self.enc1      = PartialConvBlock(in_channels, c,   stride=1)      # (B,64, H,   W  )
        self.enc1_down = PartialConvBlock(c,           c,   stride=2)      # (B,64, H/2, W/2)

        # Level 2: → (B, 128, H/2, W/2) → pool → (B, 128, H/4, W/4)
        self.enc2      = PartialConvBlock(c,    c * 2, stride=1)           # (B,128,H/2, W/2)
        self.enc2_down = PartialConvBlock(c*2,  c * 2, stride=2)           # (B,128,H/4, W/4)

        # Level 3: → (B, 256, H/4, W/4) → pool → (B, 256, H/8, W/8)
        self.enc3      = PartialConvBlock(c*2,  c * 4, stride=1)           # (B,256,H/4, W/4)
        self.enc3_down = PartialConvBlock(c*4,  c * 4, stride=2)           # (B,256,H/8, W/8)

        # Level 4: → (B, 512, H/8, W/8) — no downsample here, bottleneck next
        self.enc4      = PartialConvBlock(c*4,  c * 8, stride=1)           # (B,512,H/8, W/8)

        # ── Bottleneck ────────────────────────────────────────────────────────
        # (B, 512, H/8, W/8) → stride-2 → (B, 512, H/16, W/16)
        self.btn_down  = PartialConvBlock(c*8,  c * 8, stride=2)           # (B,512,H/16,W/16)
        self.btn       = PartialConvBlock(c*8,  c * 8, stride=1)           # (B,512,H/16,W/16)

        # ── Decoder (standard Conv with skip connections) ─────────────────────
        # Each level: upsample × 2 → cat(skip) → 2× Conv3×3

        # dec4: upsample(B,512,H/16,W/16) + skip enc4(B,512,H/8,W/8) → (B,256,H/8,W/8)
        self.dec4 = DecoderBlock(c*8, c*8, c*4)                            # (B,256,H/8, W/8)

        # dec3: upsample(B,256,H/8,W/8)  + skip enc3(B,256,H/4,W/4) → (B,128,H/4,W/4)
        self.dec3 = DecoderBlock(c*4, c*4, c*2)                            # (B,128,H/4, W/4)

        # dec2: upsample(B,128,H/4,W/4)  + skip enc2(B,128,H/2,W/2) → (B,64,H/2,W/2)
        self.dec2 = DecoderBlock(c*2, c*2, c)                              # (B,64, H/2, W/2)

        # dec1: upsample(B,64,H/2,W/2)   + skip enc1(B,64,H,W)      → (B,32,H,W)
        self.dec1 = DecoderBlock(c, c, c // 2)                             # (B,32, H,   W  )

        # ── Output head ───────────────────────────────────────────────────────
        # (B, 32, H, W) → (B, 1, H, W) in [0, 1]
        self.output_conv = nn.Sequential(
            nn.Conv2d(c // 2, 1, kernel_size=1),   # 1×1 conv
            nn.Sigmoid(),                           # binary road probability
        )

    def forward(self,
                corrupted_mask: torch.Tensor,
                hole_mask:      torch.Tensor) -> torch.Tensor:
        """
        Args:
            corrupted_mask : (B, 1, H, W) float32 — road mask with zeros in holes
            hole_mask      : (B, 1, H, W) float32 — 1=valid/known, 0=missing

        Returns:
            pred : (B, 1, H, W) float32 — completed road mask in [0, 1]
        """
        # ── Build 2-channel input ─────────────────────────────────────────────
        # (B, 1, H, W) cat (B, 1, H, W) → (B, 2, H, W)
        x    = torch.cat([corrupted_mask, hole_mask], dim=1)   # (B, 2, H,   W  )
        mask = hole_mask                                        # (B, 1, H,   W  )

        # ── Encoder ───────────────────────────────────────────────────────────
        # Level 1
        e1, m1    = self.enc1(x, mask)                         # (B, 64,  H,   W  )
        e1d, m1d  = self.enc1_down(e1, m1)                     # (B, 64,  H/2, W/2)

        # Level 2
        e2, m2    = self.enc2(e1d, m1d)                        # (B, 128, H/2, W/2)
        e2d, m2d  = self.enc2_down(e2, m2)                     # (B, 128, H/4, W/4)

        # Level 3
        e3, m3    = self.enc3(e2d, m2d)                        # (B, 256, H/4, W/4)
        e3d, m3d  = self.enc3_down(e3, m3)                     # (B, 256, H/8, W/8)

        # Level 4 (deepest encoder level, no spatial downsample)
        e4, m4    = self.enc4(e3d, m3d)                        # (B, 512, H/8, W/8)

        # ── Bottleneck ────────────────────────────────────────────────────────
        btn, mb   = self.btn_down(e4, m4)                      # (B, 512, H/16,W/16)
        btn, mb   = self.btn(btn, mb)                          # (B, 512, H/16,W/16)

        # ── Decoder ───────────────────────────────────────────────────────────
        d4 = self.dec4(btn, e4)    # upsample→H/8  + skip e4 → (B,256,H/8, W/8)
        d3 = self.dec3(d4,  e3)    # upsample→H/4  + skip e3 → (B,128,H/4, W/4)
        d2 = self.dec2(d3,  e2)    # upsample→H/2  + skip e2 → (B, 64,H/2, W/2)
        d1 = self.dec1(d2,  e1)    # upsample→H    + skip e1 → (B, 32,H,   W  )

        # ── Output ────────────────────────────────────────────────────────────
        pred = self.output_conv(d1)    # (B, 1, H, W) in [0, 1]

        return pred

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def get_inpainting_model(base_channels: int = 64) -> PartialConvUNet:
    """
    Instantiate the Partial Conv U-Net for road mask inpainting.

    Args:
        base_channels : 64 is the T4-safe default (≈ 25M parameters with AMP)
                        Reduce to 32 if you hit OOM at batch_size=8.

    Returns:
        PartialConvUNet instance (initialised, not trained)
    """
    model = PartialConvUNet(in_channels=2, base_channels=base_channels)
    n = model.count_parameters()
    print(f"🏗️  PartialConvUNet | base_channels={base_channels} | "
          f"params: {n/1e6:.2f}M")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cpu')
    model  = get_inpainting_model(base_channels=64).to(device)

    B = 2
    corrupted = torch.rand(B, 1, 512, 512)
    hole_mask = torch.ones(B, 1, 512, 512)
    hole_mask[:, :, 100:300, 150:400] = 0.0   # rectangular hole

    pred = model(corrupted, hole_mask)

    # (B, 2, 512, 512) → PartialConvUNet → (B, 1, 512, 512)
    print(f"Input  corrupted : {corrupted.shape}")
    print(f"Input  hole_mask : {hole_mask.shape}")
    print(f"Output pred      : {pred.shape}")    # expect (2, 1, 512, 512)
    assert pred.shape == (B, 1, 512, 512), "Shape mismatch!"
    assert pred.min() >= 0.0 and pred.max() <= 1.0, "Output not in [0,1]!"
    print("✅ PartialConvUNet smoke test passed")
