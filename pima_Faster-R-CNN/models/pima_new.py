import torch
import torch.nn as nn
import torch.nn.functional as F
from .vision_branch import PillDetector
from .language_branch import PrescriptionRecognizer
from .graph_branch import GraphBranch, MultiModalCrossAttention

class InfoNCELoss(nn.Module):
    """
    Computes the InfoNCE Loss for contrastive learning between visual and textual features.
    It brings embeddings of matched pairs closer while pushing unmatched ones apart.
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        import numpy as np
        # Use learnable parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, features1, features2):
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        # Limit logit_scale to avoid excessively large values
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits = torch.matmul(features1, features2.T) * logit_scale
        B = logits.shape[0]
        labels = torch.arange(B, dtype=torch.long, device=logits.device)
        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.T, labels)
        return (loss1 + loss2) / 2

class PIMA_NEW(nn.Module):
    """
    The main architecture of the Multimodal Cross-Attention Framework.
    Combines Vision Branch (Pill Detector), Language Branch (Prescription Recognizer),
    and Graph Branch (R-GAT) using Cross-Attention for pill-prescription matching.
    """
    def __init__(self, embed_dim=256, use_ocr=False, pretrained=True, vision_model='vit'):
        super(PIMA_NEW, self).__init__()
        
        self.vision_model = vision_model
        # 1. Vision Branch
        self.pill_detector = PillDetector(pretrained=pretrained, vision_model=vision_model)
        
        # Projection to common embedding space for visual features
        self.vis_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # 2. Language & OCR Branch
        self.prescription_recognizer = PrescriptionRecognizer(embed_dim=embed_dim, use_ocr=use_ocr)
        
        # 3. Graph Branch (R-GAT)
        self.graph_branch = GraphBranch(in_channels=embed_dim, hidden_channels=embed_dim//2, out_channels=embed_dim)
        
        # 4. Multi-modal Cross-Attention Fusion
        self.fusion = MultiModalCrossAttention(embed_dim=embed_dim)
        
        # 5. Losses
        self.classification_loss = nn.BCELoss()
        self.matching_loss = InfoNCELoss(temperature=0.07)
        
    def forward(self, images, pill_boxes, texts, edge_index, edge_type, matching_pairs=None, targets=None):
        # 1. Extract Visual Features
        pill_embeddings_raw, pred_bboxes, pred_scores, detection_losses = self.pill_detector(images, pill_boxes, targets)
        v_feat = self.vis_proj(pill_embeddings_raw) # [Total_Pills_in_batch, embed_dim]
        
        # 2. Extract Text Features
        flat_texts = [t for sublist in texts for t in sublist]
        t_feat_raw = self.prescription_recognizer(flat_texts) # [Total_Texts_in_batch, embed_dim]
        
        # 3. Graph Message Passing
        t_feat_graph, pill_name_probs = self.graph_branch(t_feat_raw, edge_index, edge_type)
        
        # 4. Cross Attention Fusion
        v_feat_unsq = v_feat.unsqueeze(0)
        t_feat_unsq = t_feat_graph.unsqueeze(0)
        
        fused_v_feat, attn_weights = self.fusion(v_feat_unsq, t_feat_unsq)
        fused_v_feat = fused_v_feat.squeeze(0) # [M, D]
        
        return fused_v_feat, t_feat_graph, pill_name_probs, attn_weights, detection_losses

    def compute_loss(self, fused_v_feat, t_feat_graph, pill_name_probs, true_pill_name_labels, positive_pairs, detection_losses=None):
        cls_loss = self.classification_loss(pill_name_probs.squeeze(-1), true_pill_name_labels.float())
        
        match_loss = torch.tensor(0.0, device=fused_v_feat.device, requires_grad=True)
        if len(positive_pairs) > 0:
            v_norm = F.normalize(fused_v_feat, dim=1)
            t_norm = F.normalize(t_feat_graph, dim=1)
            logit_scale = torch.clamp(self.matching_loss.logit_scale.exp(), max=100.0)
            logits = torch.matmul(v_norm, t_norm.t()) * logit_scale
            
            num_pills, num_texts = logits.shape
            target_matrix = torch.zeros_like(logits)
            for p_idx, t_idx in positive_pairs:
                if p_idx < num_pills and t_idx < num_texts:
                    target_matrix[p_idx, t_idx] = 1.0
                    
            v_mask = target_matrix.sum(dim=1) > 0
            loss_v = torch.tensor(0.0, device=logits.device)
            if v_mask.any():
                v_logits_valid = logits[v_mask]
                v_targets_valid = target_matrix[v_mask]
                v_targets_valid = v_targets_valid / v_targets_valid.sum(dim=1, keepdim=True)
                loss_v = F.cross_entropy(v_logits_valid, v_targets_valid, label_smoothing=0.1)
                
            t_mask = target_matrix.sum(dim=0) > 0
            loss_t = torch.tensor(0.0, device=logits.device)
            if t_mask.any():
                t_logits_valid = logits.t()[t_mask]
                t_targets_valid = target_matrix.t()[t_mask]
                t_targets_valid = t_targets_valid / t_targets_valid.sum(dim=1, keepdim=True)
                loss_t = F.cross_entropy(t_logits_valid, t_targets_valid, label_smoothing=0.1)
            
            match_loss = (loss_v + loss_t) / 2
            
        total_loss = match_loss + cls_loss
        
        det_loss_val = torch.tensor(0.0, device=fused_v_feat.device)
        if detection_losses is not None and len(detection_losses) > 0:
            det_loss_val = sum(loss for loss in detection_losses.values())
            total_loss = total_loss + det_loss_val
            
        return total_loss, match_loss, cls_loss, det_loss_val

if __name__ == "__main__":
    model = PIMA_NEW(vision_model='faster_rcnn')
    print("PIMA_NEW initialized with Faster R-CNN.")
