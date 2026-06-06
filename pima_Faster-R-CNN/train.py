import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from models.pima_new import PIMA_NEW
from data.dataset import PimaDataset, collate_fn
import torch.optim as optim
from tqdm import tqdm
import os
import torch.nn.utils as nn_utils
from eval import evaluate

def train_epoch(model, dataloader, optimizer, device, vision_model='vit', accumulate_steps=4):
    model.train()
    total_loss = 0
    total_match_loss = 0
    total_cls_loss = 0
    total_det_loss = 0
    
    # Try to determine if we are the main process for tqdm
    is_main = True
    if dist.is_initialized():
        is_main = dist.get_rank() == 0

    progress_bar = tqdm(dataloader, desc="Training") if is_main else dataloader
    
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(progress_bar):
        images, pill_boxes, texts, edge_indices, edge_types, true_pill_name_labels, positive_pairs = batch
        
        # Move inputs to device
        if vision_model == 'vit':
            images = [img.to(device) for img in images]
        elif vision_model == 'faster_rcnn':
            images = [img.to(device) for img in images]
            pill_boxes = [box.to(device) for box in pill_boxes]
            
        edge_indices = [e.to(device) for e in edge_indices]
        edge_types = [e.to(device) for e in edge_types]
        true_pill_name_labels = [l.to(device) for l in true_pill_name_labels]
        
        batch_loss = 0
        batch_m_loss = 0
        batch_c_loss = 0
        batch_d_loss = 0
        
        batch_v_feats = []
        batch_t_feats = []
        batch_probs = []
        batch_labels = []
        batch_det_losses = []
        global_pos_pairs = []
        
        v_offset = 0
        t_offset = 0
        
        for i in range(len(images)):
            if vision_model == 'vit' and images[i].shape[0] == 0:
                continue
            if len(texts[i]) == 0:
                continue
                
            if vision_model == 'vit':
                img = images[i] # [Num_Pills, 3, 224, 224]
                box = None
                target = None
            elif vision_model == 'faster_rcnn':
                img = [images[i]] # List of [3, H, W]
                box = [pill_boxes[i]] # List of [Num_Pills, 4]
                # Prepare targets for detection training
                target = [{'boxes': pill_boxes[i], 'labels': torch.ones(len(pill_boxes[i]), dtype=torch.int64, device=device)}]
                
            txt = [texts[i]]
            e_idx = edge_indices[i]
            e_type = edge_types[i]
            t_label = true_pill_name_labels[i]
            p_pairs = positive_pairs[i]
            
            fused_v_feat, t_feat_graph, pill_name_probs, attn_weights, det_losses = model(
                img, box, txt, e_idx, e_type, targets=target
            )
            
            batch_v_feats.append(fused_v_feat)
            batch_t_feats.append(t_feat_graph)
            batch_probs.append(pill_name_probs)
            batch_labels.append(t_label)
            if det_losses:
                batch_det_losses.append(det_losses)
            
            for p_idx, t_idx in p_pairs:
                global_pos_pairs.append((p_idx + v_offset, t_idx + t_offset))
                
            v_offset += fused_v_feat.size(0)
            t_offset += t_feat_graph.size(0)
            
        if len(batch_v_feats) > 0:
            all_v_feats = torch.cat(batch_v_feats, dim=0)
            all_t_feats = torch.cat(batch_t_feats, dim=0)
            all_probs = torch.cat(batch_probs, dim=0)
            all_labels = torch.cat(batch_labels, dim=0)
            
            # Aggregate detection losses if any
            agg_det_losses = {}
            if len(batch_det_losses) > 0:
                for dl in batch_det_losses:
                    for k, v in dl.items():
                        if k not in agg_det_losses:
                            agg_det_losses[k] = v
                        else:
                            agg_det_losses[k] += v
                for k in agg_det_losses:
                    agg_det_losses[k] /= len(batch_det_losses)
                    
            loss, m_loss, c_loss, d_loss = model.compute_loss(
                all_v_feats, all_t_feats, all_probs, all_labels, global_pos_pairs, agg_det_losses
            )
            
            batch_loss = loss
            batch_m_loss = m_loss
            batch_c_loss = c_loss
            batch_d_loss = d_loss
            
        if isinstance(batch_loss, torch.Tensor):
            batch_loss = batch_loss / len(images)
            (batch_loss / accumulate_steps).backward()
            
            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(dataloader):
                if dist.is_initialized():
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= dist.get_world_size()
                            
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += batch_loss.item()
            total_match_loss += (batch_m_loss / len(images)).item()
            total_cls_loss += (batch_c_loss / len(images)).item()
            if isinstance(batch_d_loss, torch.Tensor):
                total_det_loss += (batch_d_loss / len(images)).item()
            
        if is_main and isinstance(batch_loss, torch.Tensor):
            progress_bar.set_postfix({'loss': batch_loss.item(), 'match': (batch_m_loss/len(images)).item(), 'det': (batch_d_loss/len(images)).item() if isinstance(batch_d_loss, torch.Tensor) else 0})
            
    return total_loss / len(dataloader), total_match_loss / len(dataloader), total_cls_loss / len(dataloader), total_det_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=2) # Default 2 for Faster R-CNN
    parser.add_argument('--accumulate-steps', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--vision-model', type=str, default='vit', choices=['vit', 'faster_rcnn'])
    parser.add_argument('--save-name', type=str, default='model_best.pth')
    # DDP args
    parser.add_argument('--local-rank', type=int, default=os.environ.get("LOCAL_RANK", -1))
    args = parser.parse_args()
    
    local_rank = args.local_rank
    is_ddp = local_rank != -1
    
    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    train_dataset = PimaDataset(split_file='splits/train_scenario_1-1.json', data_dir='/home/lab/son/lap/vaipepill2022/public_train', is_train=True, vision_model=args.vision_model)
    
    val_dataset = PimaDataset(split_file='splits/test_scenario_1-1.json', data_dir='/home/lab/son/lap/vaipepill2022/public_train', is_train=False, vision_model=args.vision_model)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    sampler = DistributedSampler(train_dataset) if is_ddp else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), 
                              collate_fn=collate_fn, num_workers=4, sampler=sampler)
                              
    model = PIMA_NEW(vision_model=args.vision_model).to(device)
    
    if is_ddp:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)
            
        t_loss, m_loss, c_loss, d_loss = train_epoch(model, train_loader, optimizer, device, vision_model=args.vision_model, accumulate_steps=args.accumulate_steps)
        scheduler.step()
        
        if (not is_ddp) or (is_ddp and dist.get_rank() == 0):
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {t_loss:.4f} - Match: {m_loss:.4f} - Cls: {c_loss:.4f} - Det: {d_loss:.4f}")
            
            # Evaluate on validation set
            acc, correct, total = evaluate(model, val_loader, device, vision_model=args.vision_model)
            print(f"Validation Accuracy: {acc*100:.2f}% ({correct}/{total})")
            
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                os.makedirs('logs/weights', exist_ok=True)
                save_path = f'logs/weights/{args.save_name}'
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path} with Accuracy {acc*100:.2f}%")
            else:
                patience_counter += 1
                print(f"EarlyStopping counter: {patience_counter} out of {args.patience}")
                
        # DDP synchronization for early stopping
        if is_ddp:
            # Need to broadcast patience_counter so all GPUs stop at the same time
            stop_tensor = torch.tensor([1 if patience_counter >= args.patience else 0], dtype=torch.int, device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                if dist.get_rank() == 0:
                    print("Early stopping triggered across all GPUs!")
                break
        else:
            if patience_counter >= args.patience:
                print("Early stopping triggered!")
                break

    # ADDED: Ensure all GPUs finish the entire epoch before disconnecting
    if is_ddp:
        dist.barrier()  
        dist.destroy_process_group()

if __name__ == "__main__":
    main()