from torch import nn
import torch.nn.functional as F
from models import ImageEncoder, ProjectionHead, ImageEncoderTimm, sentencesTransformer, SBERTxSAGE
import torch


class PrescriptionPill(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=args.image_model_name,
                                              pretrained=args.image_pretrained, trainable=args.image_trainable)

        self.graph_encoder = SBERTxSAGE(
            input_dim=args.projection_dim, output_dim=args.graph_embedding, dropout_rate=args.dropout)

        self.use_ocr = getattr(args, 'use_ocr', False)
        
        if self.use_ocr:
            self.modality_encoder = sentencesTransformer(model_name=args.text_model_name,
                                                   trainable=args.text_trainable)
            self.modality_projection = ProjectionHead(
                embedding_dim=args.text_embedding, projection_dim=args.projection_dim, dropout=args.dropout)
        else:
            self.modality_encoder = ImageEncoder(model_name=args.image_model_name,
                                                  pretrained=args.image_pretrained, trainable=args.image_trainable)
            self.modality_projection = ProjectionHead(
                embedding_dim=args.image_embedding, projection_dim=args.projection_dim, dropout=args.dropout)
                
        self.image_projection = ProjectionHead(
            embedding_dim=args.image_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.graph_projection = ProjectionHead(
            embedding_dim=args.graph_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.post_process_layers = nn.Sequential(
            nn.BatchNorm1d(256, affine=False),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, 2),
            nn.GELU()
        )
                        
    def forward_graph(self, data, sentences_feature):
        # Getting graph embedding
        graph_features = self.graph_encoder(data, sentences_feature)
        
        # Prevent BatchNorm1d error when N=1
        if graph_features.size(0) == 1:
            self.post_process_layers[0].eval()
            graph_extract = self.post_process_layers(graph_features)
            self.post_process_layers[0].train(self.training)
        else:
            graph_extract = self.post_process_layers(graph_features)
            
        graph_extract = F.log_softmax(graph_extract, dim=-1)        
        return graph_extract

    def get_image_aggregation(self, image):
        x = self.image_encoder(image)
        x = self.image_projection(x)
        return x


    def forward(self, data):
        # IMAGE
        image_aggregation = self.get_image_aggregation(data.pills_images)
        
        # TEXT or BOX IMAGE (Depending on use_ocr)
        if self.use_ocr:
            modality_feature = self.modality_encoder(
                data.text_sentences_ids, data.text_sentences_mask)
        else:
            modality_feature = self.modality_encoder(data.pres_box_images)
            
        modality_projection = self.modality_projection(modality_feature)

        # GRAPH
        graph_extract = self.forward_graph(data, modality_projection)
        
        return image_aggregation, modality_projection, graph_extract
