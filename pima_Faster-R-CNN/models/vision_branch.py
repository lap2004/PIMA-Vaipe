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
    def __init__(self, pretrained=True, vision_model='vit'):
        super(PillDetector, self).__init__()
        self.vision_model = vision_model
        
        if self.vision_model == 'vit':
            # 1. ViT Backbone
            self.backbone = ViTBackbone(pretrained=pretrained)
            self.fc = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256)
            )
        elif self.vision_model == 'faster_rcnn':
            # 2. Faster R-CNN Backbone
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            self.faster_rcnn = fasterrcnn_resnet50_fpn(weights=weights)
            
            # Use Faster R-CNN Box Head features (output is 1024)
            self.fc = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256)
            )
        
    def forward(self, images, boxes=None, targets=None):
        """
        If vision_model == 'vit':
            images: [Total_Pills_in_batch, 3, 224, 224] - cropped images
            boxes: unused
        If vision_model == 'faster_rcnn':
            images: List[Tensor] or [B, 3, H, W] - full images
            boxes: List[Tensor] - ground truth boxes for extraction (shape: [Num_Pills, 4])
            targets: List[Dict] - for training Faster R-CNN detection losses
        """
        detection_losses = {}
        
        if self.vision_model == 'vit':
            features = self.backbone(images) # [Total_Pills, 768, 14, 14]
            flat_features = torch.mean(features, dim=[2, 3]) # [Total_Pills, 768]
            pill_embeddings = self.fc(flat_features)
            return pill_embeddings, None, None, detection_losses
            
        elif self.vision_model == 'faster_rcnn':
            # 1. Extract Backbone features
            # images should be a list of tensors [3, H, W]
            if isinstance(images, torch.Tensor):
                images_list = list(images)
            else:
                images_list = images
                
            # If training targets are provided, compute detection losses
            if self.training and targets is not None:
                # We need to ensure FasterRCNN is in train mode to return losses
                self.faster_rcnn.train()
                detection_losses = self.faster_rcnn(images_list, targets)
            
            # To extract features for contrastive learning, we manually forward through the components
            # We can use the ground truth boxes during training, or predicted boxes during inference
            self.faster_rcnn.eval() # Switch to eval to just extract features
            with torch.no_grad():
                original_image_sizes = [img.shape[-2:] for img in images_list]
                images_transformed, targets_transformed = self.faster_rcnn.transform(images_list, targets)
                features = self.faster_rcnn.backbone(images_transformed.tensors)
                
                proposals = None
                if boxes is not None:
                    # Use provided boxes (ground truth)
                    # We need to scale boxes according to the transform
                    proposals = []
                    for b, orig_sz in zip(boxes, original_image_sizes):
                        # Simple scaling approximation
                        scale_x = images_transformed.tensors.shape[-1] / orig_sz[1]
                        scale_y = images_transformed.tensors.shape[-2] / orig_sz[0]
                        scaled_boxes = b.clone()
                        scaled_boxes[:, 0::2] *= scale_x
                        scaled_boxes[:, 1::2] *= scale_y
                        proposals.append(scaled_boxes)
                else:
                    # Predict boxes (Inference)
                    proposals, _ = self.faster_rcnn.rpn(images_transformed, features)
                    
                # Extract RoI features
                box_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals, images_transformed.image_sizes)
                box_features = self.faster_rcnn.roi_heads.box_head(box_features) # [Total_Pills, 1024]
            
            # Re-enable gradient for the projection head
            box_features = box_features.detach().requires_grad_(True)
            pill_embeddings = self.fc(box_features)
            
            # Restore training mode if needed
            if self.training:
                self.faster_rcnn.train()
                
            return pill_embeddings, proposals, None, detection_losses

if __name__ == "__main__":
    model = PillDetector(vision_model='faster_rcnn')
    x = torch.randn(2, 3, 800, 800)
    boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32), 
             torch.tensor([[5, 5, 30, 30]], dtype=torch.float32)]
    embeddings, pred_bboxes, pred_scores, losses = model(list(x), boxes)
    print("Embeddings shape:", embeddings.shape)
