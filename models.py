# =============================================================================
# models.py  –  Segmentation Models for DeepGlobe Pipeline
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  AVAILABLE MODEL FUNCTIONS                                              │
# │  ─────────────────────────────────────────────────────────────────────  │
# │  ROAD EXTRACTION (binary, 1 class)                                      │
# │    get_road_model()           DeepLabV3+ ResNet34   ~25M params         │
# │    get_road_model_heavy()     DeepLabV3+ EffNet-B3  ~45M params         │
# │                                                                         │
# │  LAND COVER (multi-class, 7 classes)                                   │
# │    get_landcover_model()      DeepLabV3+ ResNet34   ~25M params         │
# │    get_landcover_model_heavy() DeepLabV3+ EffNet-B4 ~55M params         │
# │                                                                         │
# │  BUILDING DETECTION (binary, 1 class) ← Stage 4, NEW                   │
# │    get_building_model()        UnetPlusPlus ResNet50+scse  ~35M params  │
# │    get_building_model_heavy()  UnetPlusPlus EffNet-B5+scse ~65M params  │
# │    get_building_model_hrnet()  Unet HRNet-W48+scse  ~65M params         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Stage-4 Building Detection additions (v4):
#   • get_building_model()       → UnetPlusPlus ResNet50 + scse attention
#   • get_building_model_heavy() → UnetPlusPlus EfficientNet-B5 + scse
#   • get_building_model_hrnet() → Unet HRNet-W48 + scse
#
# Previous road and landcover models unchanged — all three pipelines coexist.
# =============================================================================

import segmentation_models_pytorch as smp


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Road Extraction Models
# ─────────────────────────────────────────────────────────────────────────────

def get_road_model():
    """
    DeepLabV3+ with ResNet34 backbone for binary road segmentation.

    DeepLabV3+ uses ASPP for multi-scale context — critical for roads that
    appear at vastly different widths (urban highways vs rural dirt tracks).
    ResNet34 doubles the block count of ResNet18, preserving thin-edge detail.

    Returns:
        smp.DeepLabV3Plus — output: raw logits (B, 1, H, W)
    """
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )


def get_road_model_heavy():
    """
    DeepLabV3+ with EfficientNet-B3 backbone — use when ≥16 GB VRAM available.

    EfficientNet-B3's squeeze-excitation blocks provide channel attention that
    helps disambiguate spectrally similar land cover types (roads vs riverbeds).

    Returns:
        smp.DeepLabV3Plus — output: raw logits (B, 1, H, W)
    """
    return smp.DeepLabV3Plus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Land Cover Models (multi-class, 7 categories)
# ─────────────────────────────────────────────────────────────────────────────

def get_landcover_model():
    """
    DeepLabV3+ with ResNet34 backbone for 7-class land cover segmentation.

    encoder_output_stride=16 retains finer boundary detail vs default stride-32.
    Critical for distinguishing Forest vs Rangeland boundaries in satellite imagery.

    Returns:
        smp.DeepLabV3Plus — output: raw logits (B, 7, H, W)
    """
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        encoder_output_stride=16,
        in_channels=3,
        classes=7,
        activation=None,
    )


def get_landcover_model_heavy():
    """
    DeepLabV3+ with EfficientNet-B4 backbone — optional upgrade for 7-class.

    Use when ≥16 GB VRAM is available. EfficientNet-B4 compound scaling gives
    ResNet50-level accuracy at fewer FLOPs; channel attention helps separate
    spectrally similar classes (Forest vs Agriculture).

    Returns:
        smp.DeepLabV3Plus — output: raw logits (B, 7, H, W)
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
# Section 3 ▸ Building Detection Models (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────

def get_building_model():
    """
    UnetPlusPlus with ResNet50 backbone + scse attention for building footprint
    extraction (binary segmentation).

    Why UnetPlusPlus over plain Unet:
        Nested dense skip connections re-use intermediate feature maps at
        multiple scales — superior for objects with complex multi-scale structure
        (small rural sheds vs large warehouse footprints in the same image).
        On a single T4 (15 GB) this would be borderline; dual T4 (30 GB) makes
        it viable at batch=32, 640×640, fp16.

    Why ResNet50 over ResNet34 for buildings:
        Buildings have highly complex surface textures: glass, tile, concrete,
        green roof. ResNet50's bottleneck blocks provide richer channel-wise
        features for rooftop material discrimination vs ResNet34's basic blocks.

    Why scse (Squeeze-Channel-Spatial Excitation) attention in decoder:
        scse applies both channel and spatial recalibration at each decoder stage.
        This suppresses building-shadow false positives (shadows share spectral
        similarity with dark rooftops) and highlights high-contrast roof edges.

    Returns:
        smp.UnetPlusPlus — output: raw logits (B, 1, H, W)
    """
    return smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        in_channels=3,
        classes=1,
        activation=None,
    )


def get_building_model_heavy():
    """
    UnetPlusPlus with EfficientNet-B5 backbone + scse attention.

    EfficientNet-B5 provides the best accuracy-per-FLOP among the EfficientNet
    family for dense prediction tasks. Its compound scaling (width+depth+res)
    was tuned to maximise ImageNet accuracy — the deeper/wider feature pyramid
    is especially effective for rooftop texture diversity.

    ⚠  VRAM requirement: ~22 GB at batch=16, 640×640, fp16.
       Safe on dual T4 (30 GB total) but leaves minimal headroom.
       Recommend batch_size=16 (8 per GPU) instead of 32 for this model.

    Returns:
        smp.UnetPlusPlus — output: raw logits (B, 1, H, W)
    """
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        in_channels=3,
        classes=1,
        activation=None,
    )


def get_building_model_hrnet():
    """
    Unet with HRNet-W48 backbone + scse attention — best boundary precision.

    HRNet (High-Resolution Network) maintains FULL resolution feature maps
    throughout the entire encoder — it never downsamples to a low-resolution
    bottleneck. Instead, it runs parallel branches at 4 resolutions
    (1/1, 1/2, 1/4, 1/8) and fuses them at each stage, so high-resolution
    spatial detail is NEVER discarded.

    Why this matters for buildings:
        Buildings are rectilinear (sharp 90° corners). Standard encoders that
        downsample to 1/32 lose corner precision; HRNet's high-res branch
        preserves exact pixel boundaries — critical for IoU on small buildings
        where a 1-pixel boundary error can reduce IoU by 10%+.

    ⚠  VRAM note: HRNet-W48 requires ~2× the VRAM of ResNet50 at the same
       batch size. Use batch_size=16 (8 per GPU) on dual T4.

    Returns:
        smp.Unet — output: raw logits (B, 1, H, W)
    """
    return smp.Unet(
        encoder_name="tu-hrnet_w48",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        in_channels=3,
        classes=1,
        activation=None,
    )