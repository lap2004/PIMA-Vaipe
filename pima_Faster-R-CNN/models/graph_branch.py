import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv

class GraphBranch(nn.Module):
    """
    Models spatial relationships between text bounding boxes using Relational Graph Attention Network (R-GAT).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=4, heads=4):
        super(GraphBranch, self).__init__()
        """
        R-GAT (Relational Graph Attention Network) replacing GraphSAGE.
        num_relations: e.g., 4 relations (left, right, top, bottom or semantic relations)
        """
        # First R-GAT layer
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        # Second R-GAT layer (output dim = out_channels)
        self.conv2 = RGATConv(hidden_channels * heads, out_channels, num_relations, heads=1, concat=False)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Binary Pseudo Classifier to highlight text boxes with pill names
        self.pseudo_classifier = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_type):
        """
        x: Node features (Text Embeddings) [N, in_channels]
        edge_index: Graph connectivity [2, E]
        edge_type: Relation types for each edge [E]
        """
        # R-GAT message passing
        # alpha coefficients are computed internally in RGATConv
        x = self.conv1(x, edge_index, edge_type)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        node_embeddings = self.conv2(x, edge_index, edge_type)
        node_embeddings = self.norm2(node_embeddings)
        
        # Pseudo classification (Probability of a box containing a pill name)
        # g_i score
        pill_name_probs = self.pseudo_classifier(node_embeddings)
        
        # Highlight boxes: Multiply embedding with probability
        weighted_embeddings = node_embeddings * pill_name_probs
        
        return weighted_embeddings, pill_name_probs

class MultiModalCrossAttention(nn.Module):
    """
    Fuses visual and textual modalities using Cross-Attention, where visual features query the textual graph.
    """
    def __init__(self, embed_dim, num_heads=4):
        super(MultiModalCrossAttention, self).__init__()
        """
        Cross-Attention fusion layer.
        F_vis (Pill visual features) queries F_graph (Text graph features).
        Instead of simple cosine similarity between static vectors.
        """
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, visual_features, text_features):
        """
        visual_features: [B, M, embed_dim] (M = number of pills)
        text_features: [B, N, embed_dim] (N = number of text boxes)
        
        visual queries text to find relevant text contexts.
        """
        # Queries: Visual features, Keys/Values: Text features
        attn_output, attn_weights = self.cross_attn(query=visual_features, 
                                                    key=text_features, 
                                                    value=text_features)
        
        # Add & Norm
        out = self.norm(visual_features + attn_output)
        
        # FFN
        ffn_out = self.ffn(out)
        
        # Add & Norm
        fused_features = self.norm2(out + ffn_out)
        
        return fused_features, attn_weights

if __name__ == "__main__":
    # Test Graph Branch
    num_nodes = 10
    in_dim = 256
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    rgat = GraphBranch(in_channels=in_dim, hidden_channels=128, out_channels=256)
    out_x, probs = rgat(x, edge_index, edge_type)
    print("Graph Output shape:", out_x.shape)
    print("Probs shape:", probs.shape)
    
    # Test Cross Attention
    fusion = MultiModalCrossAttention(embed_dim=256)
    v_feat = torch.randn(2, 5, 256) # 2 images, 5 pills
    t_feat = torch.randn(2, 10, 256) # 2 images, 10 text boxes
    fused, weights = fusion(v_feat, t_feat)
    print("Fused shape:", fused.shape)
    print("Attn weights shape:", weights.shape)
