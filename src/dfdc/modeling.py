import torch.nn as nn
import torchvision.models as models

def build_model(model_name: str = "convnext_tiny", num_classes: int = 2):
    model_name = model_name.lower()

    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model, weights.transforms()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, weights.transforms()

    raise ValueError(f"Unknown model_name: {model_name}")
