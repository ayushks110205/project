import segmentation_models_pytorch as smp

def get_road_model():
    """
    Creates a DeepLabV3+ model with a ResNet18 backbone as 
    recommended by the DeepGlobe baseline paper.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",        # The "backbone"
        encoder_weights="imagenet",     # Use pre-trained weights for faster convergence
        in_channels=3,                  # RGB satellite images
        classes=1,                      # Binary output (Road vs. Not Road)
        activation=None                 # We'll use Sigmoid inside our loss function
    )
    return model

# If you want to tackle the Land Cover track later (multi-class):
def get_landcover_model():
    return smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=7,                      # 7 classes: Urban, Agriculture, etc.
        activation=None
    )

# Add this to your local models.py
def get_building_model():
    # Using ResNet34 this time could be a good "upgrade" for buildings 
    # to capture finer details, but ResNet18 works too!
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1 # Binary: Building or Not Building
    )
    return model