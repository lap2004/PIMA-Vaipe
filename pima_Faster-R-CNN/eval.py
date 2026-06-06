import torch
import argparse
from torch.utils.data import DataLoader
from models.pima_new import PIMA_NEW
from data.dataset import PimaDataset, collate_fn
from tqdm import tqdm

def evaluate(model, dataloader, device, vision_model='vit'):
    model.eval()
    
    correct_top1 = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, pill_boxes, texts, edge_indices, edge_types, true_pill_name_labels, positive_pairs = batch
            
            if vision_model == 'vit':
                images = [img.to(device) for img in images]
            elif vision_model == 'faster_rcnn':
                images = [img.to(device) for img in images]
                pill_boxes = [box.to(device) for box in pill_boxes]
                
            edge_indices = [e.to(device) for e in edge_indices]
            edge_types = [e.to(device) for e in edge_types]
            
            for i in range(len(images)):
                if vision_model == 'vit' and images[i].shape[0] == 0:
                    continue
                if len(texts[i]) == 0:
                    continue
                    
                if vision_model == 'vit':
                    img = images[i]
                    box = None
                elif vision_model == 'faster_rcnn':
                    img = [images[i]]
                    box = [pill_boxes[i]]
                    
                txt = [texts[i]]
                e_idx = edge_indices[i]
                e_type = edge_types[i]
                p_pairs = positive_pairs[i]
                
                if len(p_pairs) == 0:
                    continue
                    
                fused_v_feat, t_feat_graph, _, _, _ = model(
                    img, box, txt, e_idx, e_type
                )
                
                v_norm = torch.nn.functional.normalize(fused_v_feat, dim=1)
                t_norm = torch.nn.functional.normalize(t_feat_graph, dim=1)
                
                similarity = torch.matmul(v_norm, t_norm.t()) # [Num_Pills, Num_Texts]
                
                for p_idx, t_idx in p_pairs:
                    if p_idx < similarity.shape[0] and t_idx < similarity.shape[1]:
                        pred_text_idx = torch.argmax(similarity[p_idx]).item()
                        if pred_text_idx == t_idx:
                            correct_top1 += 1
                        total_samples += 1
                        
    accuracy = correct_top1 / max(total_samples, 1)
    return accuracy, correct_top1, total_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision-model', type=str, default='vit', choices=['vit', 'faster_rcnn'])
    parser.add_argument('--weights', type=str, default='model_best_faster_rcnn.pth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_dataset = PimaDataset(split_file='splits/test_scenario_1-1.json', data_dir='/home/lab/son/lap/vaipepill2022/public_train', is_train=False, vision_model=args.vision_model)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    model = PIMA_NEW(vision_model=args.vision_model).to(device)
    
    import os
    weight_path = f'logs/weights/{args.weights}'
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: {weight_path} not found. Evaluating with random weights.")
        
    acc, correct, total = evaluate(model, val_loader, device, vision_model=args.vision_model)
    print(f"\nEvaluation Results ({args.vision_model}):")
    print(f"Total evaluated pairs: {total}")
    print(f"Correct Top-1 Matches: {correct}")
    print(f"Accuracy: {acc*100:.2f}%")
