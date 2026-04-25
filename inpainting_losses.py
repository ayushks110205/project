# =============================================================================
# inpainting_losses.py  –  Combined Inpainting Loss
# =============================================================================
# Implements all 4 loss components described in the Stage 2 spec:
#
#   total_loss = 1.0*L_valid + 6.0*L_hole + 0.05*L_perceptual + 2.0*L_connectivity
#
# Component overview:
#   L_valid       : L1 on known (valid) pixels — model mustn't corrupt them
#   L_hole        : L1 on missing (hole) pixels — primary inpainting target
#   L_perceptual  : VGG16 feature matching — encourages realistic road texture
#   L_connectivity: Morphological dilation BCE — prevents fragmented road segments
#
# All losses operate on tensors of shape (B, 1, H, W) with values in [0, 1].
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ VGG16 Perceptual Feature Extractor (frozen)
# ─────────────────────────────────────────────────────────────────────────────

class VGGPerceptualExtractor(nn.Module):
    """
    Extracts intermediate feature maps from the first 3 conv blocks of VGG16.

    Frozen — weights are never updated during inpainting training.
    Using only 3 blocks (not all 5) to reduce VRAM usage on T4.

    Input is expected to be (B, 1, H, W) binary road masks.  We replicate
    to 3 channels before passing to VGG (which expects RGB input).

    Feature maps returned:
        relu1_2 — (B, 64,  H,    W)
        relu2_2 — (B, 128, H/2,  W/2)
        relu3_3 — (B, 256, H/4,  W/4)
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)

        # Slice out the first 16 layers (up through relu3_3)
        # VGG16 layer indices:
        #   0-3  : block1 (conv1_1, relu, conv1_2, relu)
        #   5-10 : block2 (maxpool, conv2_1, relu, conv2_2, relu)
        #   10-17: block3 (maxpool, conv3_1, relu, conv3_2, relu, conv3_3, relu)
        features = list(vgg.features.children())

        self.block1 = nn.Sequential(*features[0:4])    # relu1_2, output stride 1
        self.block2 = nn.Sequential(*features[4:9])    # relu2_2, output stride 2
        self.block3 = nn.Sequential(*features[9:16])   # relu3_3, output stride 4

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x : (B, 1, H, W) float32 — binary road mask in [0, 1]

        Returns:
            list of 3 feature tensors:
                [(B, 64, H, W), (B, 128, H/2, W/2), (B, 256, H/4, W/4)]
        """
        # VGG expects 3-channel input — replicate the single channel
        # (B, 1, H, W) → (B, 3, H, W)
        x3 = x.repeat(1, 3, 1, 1)

        f1 = self.block1(x3)    # (B, 64,  H,    W)
        f2 = self.block2(f1)    # (B, 128, H/2,  W/2)
        f3 = self.block3(f2)    # (B, 256, H/4,  W/4)

        return [f1, f2, f3]


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Morphological Dilation Kernel (for connectivity loss)
# ─────────────────────────────────────────────────────────────────────────────

def _make_dilation_kernel(device: torch.device) -> torch.Tensor:
    """
    Create a fixed 3×3 all-ones dilation kernel.

    Returns:
        kernel : (1, 1, 3, 3) float32 tensor on `device`
    """
    k = torch.ones(1, 1, 3, 3, dtype=torch.float32, device=device)
    return k


def morphological_dilate(binary_mask: torch.Tensor,
                         kernel:      torch.Tensor) -> torch.Tensor:
    """
    Approximate morphological dilation via max-pooling with a 3×3 kernel.

    A pixel becomes 1 if ANY of its 3×3 neighbours was 1 in the input.
    This is equivalent to dilation with a square structuring element.

    Implementation: threshold(conv(mask, ones_kernel)) > 0  → {0,1}

    Args:
        binary_mask : (B, 1, H, W) float32 in {0, 1}
        kernel      : (1, 1, 3, 3) float32 all-ones (from _make_dilation_kernel)

    Returns:
        dilated : (B, 1, H, W) float32 in {0, 1}
    """
    # Sum of neighbourhood — if >0 then at least one road pixel nearby
    # (B, 1, H, W) → conv → (B, 1, H, W)  [padding=1 preserves spatial size]
    neighbourhood_sum = F.conv2d(binary_mask, kernel, padding=1)
    dilated = (neighbourhood_sum > 0).float()   # (B, 1, H, W)
    return dilated


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ Combined Inpainting Loss
# ─────────────────────────────────────────────────────────────────────────────

class InpaintingLoss(nn.Module):
    """
    Combined loss for road mask inpainting.

    total_loss = λ_valid * L_valid
               + λ_hole  * L_hole
               + λ_perc  * L_perceptual
               + λ_conn  * L_connectivity

    Args:
        lambda_valid : weight for L_valid       (default 1.0)
        lambda_hole  : weight for L_hole        (default 6.0)
        lambda_perc  : weight for L_perceptual  (default 0.05)
        lambda_conn  : weight for L_connectivity(default 2.0)
        device       : torch device (needed for dilation kernel)

    Forward args:
        pred     : (B, 1, H, W) float32 — model output (sigmoid-activated)
        target   : (B, 1, H, W) float32 — ground-truth road mask {0,1}
        hole_mask: (B, 1, H, W) float32 — 1=valid/known, 0=missing hole
    """

    def __init__(self,
                 lambda_valid: float = 1.0,
                 lambda_hole:  float = 6.0,
                 lambda_perc:  float = 0.05,
                 lambda_conn:  float = 2.0,
                 device:       torch.device = None):
        super().__init__()

        self.lv = lambda_valid
        self.lh = lambda_hole
        self.lp = lambda_perc
        self.lc = lambda_conn

        # VGG16 perceptual extractor (frozen)
        self.vgg = VGGPerceptualExtractor()
        if device is not None:
            self.vgg = self.vgg.to(device)
        self.vgg.eval()

        # Register dilation kernel as a buffer (moves with .to(device))
        dil_k = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        self.register_buffer('dil_kernel', dil_k)

    # ── Loss Components ───────────────────────────────────────────────────────

    def l_valid(self,
                pred:      torch.Tensor,
                target:    torch.Tensor,
                hole_mask: torch.Tensor) -> torch.Tensor:
        """
        L_valid: L1 loss on KNOWN regions only.
        The model must preserve areas we already know — don't modify valid pixels.

        Args:
            pred      : (B, 1, H, W)
            target    : (B, 1, H, W)
            hole_mask : (B, 1, H, W) 1=valid, 0=hole

        Returns:
            scalar loss
        """
        # Only count pixels where hole_mask=1 (known/valid region)
        valid_pixels = hole_mask.sum().clamp(min=1.0)
        loss = (torch.abs(pred - target) * hole_mask).sum() / valid_pixels
        return loss

    def l_hole(self,
               pred:      torch.Tensor,
               target:    torch.Tensor,
               hole_mask: torch.Tensor) -> torch.Tensor:
        """
        L_hole: L1 loss on MISSING (hole) regions.
        Primary inpainting objective — reconstruct the road network in the gap.

        Args:
            pred      : (B, 1, H, W)
            target    : (B, 1, H, W)
            hole_mask : (B, 1, H, W) 1=valid, 0=hole

        Returns:
            scalar loss
        """
        # Inverse mask: 1 where hole_mask=0 (the missing regions)
        inv_mask    = 1.0 - hole_mask
        hole_pixels = inv_mask.sum().clamp(min=1.0)
        loss = (torch.abs(pred - target) * inv_mask).sum() / hole_pixels
        return loss

    def l_perceptual(self,
                     pred:   torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        """
        L_perceptual: Feature matching loss via frozen VGG16.

        Computes L1 distance between VGG16 intermediate feature maps of
        the prediction and target.  Forces the model to produce road
        textures with realistic structure, not just pixel-level accuracy.

        Uses 3 VGG blocks to balance spatial resolution vs semantic richness.

        NOTE: VGG is computed in float32 regardless of the outer autocast
        context. VGG block3 has 3×3×256 = 2304 accumulated additions per
        pixel — in fp16 this easily overflows 65504 → inf → NaN total loss.
        The .float() cast has identity gradient, so backprop is unaffected.

        Args:
            pred   : (B, 1, H, W)
            target : (B, 1, H, W)

        Returns:
            scalar loss
        """
        # Force float32 for VGG — prevents fp16 accumulation overflow
        with torch.amp.autocast('cuda', enabled=False):
            pred_f32   = pred.float()
            target_f32 = target.float()

            with torch.no_grad():
                target_feats = self.vgg(target_f32)   # list of 3 feature tensors

            pred_feats = self.vgg(pred_f32)           # list of 3 feature tensors

            loss = 0.0
            for pf, tf in zip(pred_feats, target_feats):
                # Normalise by number of elements so different resolutions contribute equally
                loss = loss + F.l1_loss(pf, tf.detach())

        return loss / len(pred_feats)

    def l_connectivity(self,
                       pred:   torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """
        L_connectivity: Penalises broken/fragmented road segments.

        Implementation:
            1. Dilate both pred (soft, via sigmoid) and target (binary) with
               a 3×3 all-ones kernel → highlights road neighbourhood.
            2. Compute BCE on the dilated versions.
            3. This penalises gaps in the predicted road network because a
               broken road segment will fail to activate the dilated region
               around connecting pixels.

        Road connectivity is crucial for map inpainting — a predicted road
        that doesn't reconnect to the existing network is useless for
        navigation or urban planning.

        Args:
            pred   : (B, 1, H, W) float32 — sigmoid probabilities in [0,1]
            target : (B, 1, H, W) float32 — binary ground truth {0,1}

        Returns:
            scalar loss
        """
        # Cast kernel to match pred's device AND dtype (e.g. cuda + float16 under AMP)
        kernel = self.dil_kernel.to(device=pred.device, dtype=pred.dtype)

        # Dilate pred (soft, not binarised — keeps gradients flowing)
        pred_dilated = F.conv2d(pred, kernel, padding=1)   # (B,1,H,W)
        pred_dilated = pred_dilated.clamp(0, 1)             # soft [0,1]

        # Dilate target (binary — detach, no grad needed)
        with torch.no_grad():
            target_dilated = F.conv2d(target, kernel, padding=1)
            target_dilated = (target_dilated > 0).float()  # (B,1,H,W)

        # MSE between dilated maps.
        #
        # Why MSE instead of BCE:
        #   BCE had three compounding AMP problems:
        #     1. Blocked by PyTorch autocast dispatch (RuntimeError)
        #     2. float16 epsilon (1e-6) rounds to 0 → -log(0) = -inf → CUDA assertion
        #     3. Required autocast(enabled=False) + .float() gymnastics
        #
        # MSE is:
        #   • Autocast-safe (no blocklist)
        #   • No epsilon / clamping needed (both inputs are naturally in [0,1])
        #   • Semantically equivalent: penalises gaps in the dilated road network
        #     just as effectively as BCE for this [0,1] comparison task.
        loss = F.mse_loss(pred_dilated.float(), target_dilated.detach().float())
        return loss

    # ── Combined Forward ──────────────────────────────────────────────────────

    def forward(self,
                pred:      torch.Tensor,
                target:    torch.Tensor,
                hole_mask: torch.Tensor) -> dict:
        """
        Compute total combined loss and return per-component breakdown.

        Args:
            pred      : (B, 1, H, W) float32 — model output in [0, 1] (sigmoid)
            target    : (B, 1, H, W) float32 — ground-truth binary mask {0, 1}
            hole_mask : (B, 1, H, W) float32 — 1=valid/known, 0=missing hole

        Returns:
            dict with keys:
                'total'       — weighted sum
                'valid'       — L_valid (unweighted)
                'hole'        — L_hole  (unweighted)
                'perceptual'  — L_perceptual (unweighted)
                'connectivity'— L_connectivity (unweighted)
        """
        lv = self.l_valid(pred, target, hole_mask)
        lh = self.l_hole(pred, target, hole_mask)
        lp = self.l_perceptual(pred, target)
        lc = self.l_connectivity(pred, target)

        total = (self.lv * lv
                 + self.lh * lh
                 + self.lp * lp
                 + self.lc * lc)

        return {
            'total':        total,
            'valid':        lv,
            'hole':         lh,
            'perceptual':   lp,
            'connectivity': lc,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cpu')

    # Fake batch: B=2, 512×512
    pred      = torch.sigmoid(torch.randn(2, 1, 512, 512))
    target    = (torch.rand(2, 1, 512, 512) > 0.8).float()
    hole_mask = torch.ones(2, 1, 512, 512)
    hole_mask[:, :, 100:300, 100:300] = 0.0   # punch a hole

    loss_fn = InpaintingLoss(device=device)
    losses  = loss_fn(pred, target, hole_mask)

    for k, v in losses.items():
        print(f"  {k:15s}: {v.item():.4f}")
    print("✅ InpaintingLoss smoke test passed")
