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

def train_epoch(model, dataloader, optimizer, device):
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
    for batch in progress_bar:
        images, pill_boxes, texts, edge_indices, edge_types, true_pill_name_labels, positive_pairs = batch
        
        # Move inputs to device
        images = [img.to(device) for img in images]
            
        edge_indices = [e.to(device) for e in edge_indices]
        edge_types = [e.to(device) for e in edge_types]
        true_pill_name_labels = [l.to(device) for l in true_pill_name_labels]
        
        optimizer.zero_grad()
        
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
            if images[i].shape[0] == 0:
                continue
            if len(texts[i]) == 0:
                continue
                
            img = images[i] # [Num_Pills, 3, 224, 224]
            box = None
            target = None
                
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
            
            loss_module = model.module if hasattr(model, 'module') else model
            
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
                    
            loss, m_loss, c_loss, d_loss = loss_module.compute_loss(
                all_v_feats, all_t_feats, all_probs, all_labels, global_pos_pairs, agg_det_losses
            )
            
            batch_loss = loss
            batch_m_loss = m_loss
            batch_c_loss = c_loss
            batch_d_loss = d_loss
            
        if isinstance(batch_loss, torch.Tensor):
            batch_loss = batch_loss / len(images)
            batch_loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=2)
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
        
    train_dataset = PimaDataset(split_file='splits/train_scenario_1-1.json', data_dir='/home/lab/son/lap/vaipepill2022/public_train', is_train=True)
    
    sampler = DistributedSampler(train_dataset) if is_ddp else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), 
                              collate_fn=collate_fn, num_workers=4, sampler=sampler)
                              
    model = PIMA_NEW().to(device)
    
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)
            
        t_loss, m_loss, c_loss, d_loss = train_epoch(model, train_loader, optimizer, device)
        
        if (not is_ddp) or (is_ddp and dist.get_rank() == 0):
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {t_loss:.4f} - Match: {m_loss:.4f} - Cls: {c_loss:.4f} - Det: {d_loss:.4f}")
            
            if t_loss < best_loss:
                best_loss = t_loss
                os.makedirs('logs/weights', exist_ok=True)
                save_path = f'logs/weights/{args.save_name}'
                model_to_save = model.module if is_ddp else model
                torch.save(model_to_save.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
