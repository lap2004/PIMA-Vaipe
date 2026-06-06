import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import roi_align

class ViTBackbone(nn.Module):
    """
    Vision Transformer (ViT) backbone for feature extraction.
    """
    def __init__(self, pretrained=True):
        super(ViTBackbone, self).__init__()
        vit = torchvision.models.vit_b_16(weights='DEFAULT' if pretrained else None)
        self.backbone = vit
        self.out_channels = 768
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.backbone._process_input(x)
        n = x.shape[0]
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x)
        
        features = x[:, 1:]
        spatial_size = int(features.shape[1] ** 0.5)
        features = features.transpose(1, 2).reshape(B, self.out_channels, spatial_size, spatial_size)
        return features

class PillDetector(nn.Module):
    """
    Extracts visual features from pill images using a backbone network.
    """
    def __init__(self, pretrained=True):
        super(PillDetector, self).__init__()
        
        # 1. ViT Backbone
        self.backbone = ViTBackbone(pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        
    def forward(self, images, boxes=None, targets=None):
        """
        images: [Total_Pills_in_batch, 3, 224, 224] - cropped images
        boxes: unused
        """
        detection_losses = {}
        
        features = self.backbone(images) # [Total_Pills, 768, 14, 14]
        flat_features = torch.mean(features, dim=[2, 3]) # [Total_Pills, 768]
        pill_embeddings = self.fc(flat_features)
        return pill_embeddings, None, None, detection_losses

if __name__ == "__main__":
    model = PillDetector()
    x = torch.randn(2, 3, 224, 224)
    embeddings, pred_bboxes, pred_scores, losses = model(x)
    print("Embeddings shape:", embeddings.shape)
