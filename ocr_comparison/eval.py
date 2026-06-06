import glob
import torch
from tqdm import tqdm
from models.prescription_pill import PrescriptionPill
from utils.utils import build_loaders
from utils.option import option
import warnings
warnings.filterwarnings("ignore")

def evaluate():
    print(">>>> Preparing data...")
    args = option()
    # Ensure args are set correctly for evaluation
    args.data_folder = "/home/lab/son/lap/vaipe_pima_format/"
    args.val_folder = "prescriptions/test/"
    args.val_batch_size = 1
    
    val_files = glob.glob(args.data_folder + args.val_folder + "*.json")
    val_loader = build_loaders(val_files, mode="test", batch_size=args.val_batch_size, args=args)

    print("Val files: ", len(val_files))

    print(">>>> Preparing model...")
    model = PrescriptionPill(args).cuda()
    
    if getattr(args, 'use_ocr', False):
        model_path = "./logs/weights/model_best_ocr.pth"
    else:
        model_path = "./logs/weights/model_best_no_ocr.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded weights from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")

    model.eval()
    matching_acc = []
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Evaluating"):
            data = data.cuda()

            correct = []
            image_aggregation, modality_projection, graph_extract = model(data)

            _, max_graph_extract = torch.max(graph_extract, 1)
            drugname_indices = torch.where(max_graph_extract == 0)[0]
            
            if len(drugname_indices) == 0:
                correct.append(0)
                matching_acc.append(0)
                continue
                
            modality_embedding_drugname = modality_projection[drugname_indices]
            labels_drugname = data.pills_label[drugname_indices]

            # For Matching
            image_norm = torch.nn.functional.normalize(image_aggregation, p=2, dim=1)
            text_norm = torch.nn.functional.normalize(modality_embedding_drugname, p=2, dim=1)
            similarity = image_norm @ text_norm.t()
            
            _, predicted = torch.max(similarity, 1)
            mapping_predicted = labels_drugname[predicted]

            correct.append(mapping_predicted.eq(
                data.pills_images_labels).sum().item() / len(data.pills_images_labels))

            matching_acc.append(sum(correct) / len(correct))

    if len(matching_acc) > 0:
        final_accuracy = (sum(matching_acc) / len(matching_acc)) * 100
        res = f"\n--- EVALUATION RESULTS ---\n"
        res += f"Top-1 Matching Accuracy: {final_accuracy:.2f}%\n"
        print(res)
        with open("log_eval.log", "a", encoding="utf-8") as f:
            f.write(res)
    else:
        print("No evaluation data found.")

if __name__ == '__main__':
    evaluate()
