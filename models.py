# =============================================================================
# models.py  –  Segmentation Models for DeepGlobe Pipeline
# =============================================================================
# KEY CHANGES vs v1:
#   • get_road_model()          → backbone upgraded ResNet18 → ResNet34
#   • get_road_model_heavy()    → EfficientNet-B3 (optional, >16 GB VRAM)
#   • get_landcover_model()     → upgraded ResNet18 → ResNet34 (v3)
#                                  + encoder_output_stride=16 for boundaries
#   • get_landcover_model_heavy() → EfficientNet-B4 (new, v3)
#   • get_building_model()      → unchanged (U-Net ResNet18, binary)
#
# Why ResNet34 over ResNet18 for road extraction?
# ─────────────────────────────────────────────────
# Roads are THIN linear structures (often 2-5 px wide at 512-px resolution).
# ResNet18 has only 8 residual blocks; its feature maps lose fine spatial
# detail rapidly through stride-2 downsampling.  ResNet34 doubles the
# block count (16 blocks) without the bottleneck complexity of ResNet50,
# giving richer intermediate feature maps that preserve thin-edge context —
# critical for distinguishing narrow roads from similarly-coloured terrain.
# The parameter increase (~21M → ~25M) is well within T4 VRAM budget when
# combined with AMP (fp16) training.
#
# Why EfficientNet-B3 for the "heavy" model?
# ─────────────────────────────────────────────
# EfficientNet-B3 uses compound scaling (width + depth + resolution) to
# achieve ResNet50-level accuracy at ~40% fewer FLOPs. Its squeeze-excitation
# blocks learn channel-wise attention, particularly effective for separating
# spectrally similar classes (roads vs dry riverbeds, light concrete vs sand).
# Recommended when ≥16 GB VRAM is available or when using gradient
# checkpointing + AMP on a Colab Pro / A100 instance.
# =============================================================================

import segmentation_models_pytorch as smp


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Road Extraction Models
# ─────────────────────────────────────────────────────────────────────────────

def get_road_model():
    """
    DeepLabV3+ with ResNet34 backbone for binary road segmentation.

    Architecture choice:
        DeepLabV3+ uses Atrous Spatial Pyramid Pooling (ASPP) to capture
        multi-scale context at multiple dilation rates simultaneously.
        This is especially valuable for roads, which appear at vastly
        different widths (urban highways vs. rural dirt tracks) in the
        same image.  The decoder's skip-connections from the ResNet34
        low-level features preserve thin-edge sharpness.

    Returns:
        smp.DeepLabV3Plus model (output: raw logits, shape [B, 1, H, W])
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",        # Upgraded from resnet18 for richer features
        encoder_weights="imagenet",     # ImageNet pre-training for fast convergence
        in_channels=3,                  # RGB satellite images
        classes=1,                      # Binary: road (1) vs background (0)
        activation=None,                # Raw logits — Sigmoid applied in loss/eval
    )
    return model


def get_road_model_heavy():
    """
    DeepLabV3+ with EfficientNet-B3 backbone — optional upgrade.

    Use this when ≥16 GB VRAM is available (Colab Pro, A100, L4).
    EfficientNet-B3's squeeze-excitation blocks provide channel attention
    that helps disambiguate spectrally similar land cover types.

    ⚠  On free-tier T4 (15 GB), use with AMP (fp16) + batch_size=8.
       Prefer get_road_model() for standard T4 training at batch_size=16.

    Returns:
        smp.DeepLabV3Plus model (output: raw logits, shape [B, 1, H, W])
    """
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Land Cover Model (multi-class, 7 categories)
# ─────────────────────────────────────────────────────────────────────────────

def get_landcover_model():
    """
    DeepLabV3+ with ResNet34 backbone for 7-class land cover segmentation.

    # Multi-class segmentation needs deeper backbone than binary tasks —
    # resnet34 captures vegetation texture better than resnet18.

    Architecture notes:
      • ResNet34 (vs ResNet18): 16 vs 8 residual blocks — richer intermediate
        features for distinguishing spectrally similar classes like Forest vs
        Rangeland (both green, differ in texture and saturation).
      • encoder_output_stride=16: default stride-32 loses 2× spatial resolution
        in the encoder. Stride-16 retains finer boundary detail at the cost of
        slightly more VRAM — well within T4 budget at batch_size=8 + AMP.
      • activation=None: raw logits — Softmax is fused into the loss function.

    Returns:
        smp.DeepLabV3Plus model (output: raw logits, shape [B, 7, H, W])
        # (B, 3, 512, 512) → encoder → (B, 256, 32, 32) → ASPP → decoder
        # → (B, 7, 512, 512) logits → argmax(dim=1) → (B, 512, 512) class IDs
    """
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",          # Upgraded: richer vegetation texture
        encoder_weights="imagenet",
        encoder_output_stride=16,         # Better boundary detail vs default 32
        in_channels=3,
        classes=7,
        activation=None,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 2b ▸ Land Cover Heavy Model (optional upgrade)
# ─────────────────────────────────────────────────────────────────────────────

def get_landcover_model_heavy():
    """
    DeepLabV3+ with EfficientNet-B4 backbone — optional upgrade for 7-class.

    Use when ≥16 GB VRAM is available (Colab Pro, A100, L4).
    EfficientNet-B4 uses compound scaling (width+depth+resolution) giving
    ResNet50-level accuracy at fewer FLOPs; its squeeze-excitation blocks
    attend channel-wise, effective for separating spectrally similar classes.

    ⚠  On free-tier T4 (15 GB), use get_landcover_model() at batch_size=8+AMP.

    Returns:
        smp.DeepLabV3Plus model (output: raw logits, shape [B, 7, H, W])
        # (B, 3, 512, 512) → encoder → (B, 448, 32, 32) → ASPP → decoder
        # → (B, 7, 512, 512) logits
    """
    return smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        encoder_output_stride=16,
        in_channels=3,
        classes=7,
        activation=None,
    )


# ─────────────────────────────────────────────────────────────────────────────

def get_building_model():
    """
    U-Net with ResNet18 backbone for binary building footprint extraction.

    U-Net is preferred over DeepLabV3+ here because buildings are compact,
    roughly rectangular objects — U-Net's symmetric encoder-decoder with
    dense skip connections excels at preserving object boundaries without
    the heavy ASPP overhead.

    Returns:
        smp.Unet model (output: raw logits, shape [B, 1, H, W])
    """
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,      # Binary: building (1) vs background (0)
        activation=None,
    )