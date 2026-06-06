import glob
import os
import torch
from tqdm import tqdm
from models.prescription_pill import PrescriptionPill
from data_inference.data_inference import PrescriptionPillData
from torch_geometric.data import DataLoader
from utils.option import option
from torch import nn

def build_loaders(files, batch_size=1, args=None):
    dataset = PrescriptionPillData(files, args)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True)
    return dataloader

def inference(args):
    inference_dir = getattr(args, 'inference_folder', 'data_inference/data/pres/')
    model_path   = getattr(args, 'model_path', os.path.join(args.save_folder, 'model_best.pth'))

    inference_file = glob.glob(inference_dir + "*.json")
    inference_loaders = build_loaders(inference_file, batch_size=args.val_batch_size, args=args)
    
    model = PrescriptionPill(args).cuda()
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    for data in inference_loaders:
        data = data.cuda()
        
        print(data.pills_labels)
        image_aggregation, sentences_projection, graph_extract = model(data)
        for image in image_aggregation:
            similarity = cos(image, sentences_projection)       
            max_sim, predicted = torch.max(similarity, 0)
            
            # Ngưỡng similarity: nếu không có thuốc nào đạt > 0.5 thì là No match
            if max_sim.item() < 0.5:
                print("Predicted: No match (max similarity = {:.4f})".format(max_sim.item()))
            else:
                print("Predicted: ", data.text[0][predicted.item()], "(similarity = {:.4f})".format(max_sim.item()))
            
if __name__ == '__main__':
    parse_args = option()
    inference(parse_args)
