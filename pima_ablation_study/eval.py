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
    
    model_path = "./run/weights/model_best.pth"
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
            image_aggregation, sentences_projection, graph_extract = model(data)

            # For Matching
            similarity_text_matching = image_aggregation @ sentences_projection.t() 
            similarity_text_matching = torch.nn.functional.softmax(similarity_text_matching, dim=1)
            similarity = torch.where(similarity_text_matching > 0.8, similarity_text_matching, torch.zeros_like(similarity_text_matching))
            
            _, predicted = torch.max(similarity, 1)
            mapping_predicted = data.pills_label[predicted]

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
