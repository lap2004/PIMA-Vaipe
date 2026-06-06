import glob
import torch
from tqdm import tqdm
from models.prescription_pill import PrescriptionPill
from utils.metrics import ContrastiveLoss, TripletLoss
import wandb
from utils.utils import build_loaders, calculate_matching_loss
from utils.option import option
import warnings
import config as CFG
warnings.filterwarnings("ignore")


def train(model, train_loader, optimizer, matching_criterion, graph_criterion, epoch):
    model.train()
    train_loss = []
    wandb.watch(model)
    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        # Loop through for each prescription
        for data in train_bar:
            data = data.cuda()
            optimizer.zero_grad()
            pre_loss = []

            image_aggregation, modality_projection, graph_extract = model(data)
            _, max_graph_extract = torch.max(graph_extract, 1)

            # Create for Image matching Drugname
            modality_embedding_drugname = modality_projection[max_graph_extract == 0]
            sentences_labels_drugname = data.pills_label[max_graph_extract == 0]
            
            matching_loss = calculate_matching_loss(image_aggregation, modality_embedding_drugname, sentences_labels_drugname, data.pills_images_labels, matching_criterion)

            # Create for Image matching Graph
            graph_loss = graph_criterion(graph_extract, data.y)
            loss = matching_loss + graph_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pre_loss.append(loss.item())

            train_loss.append(sum(pre_loss) / len(pre_loss))
            train_bar.set_postfix(loss=train_loss[-1])

    return sum(train_loss) / len(train_loss)


def val(model, val_loader):
    """
    Summary: Test with all text embedding
    """
    model.eval()
    matching_acc = []
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
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

    final_accuracy = sum(matching_acc) / len(matching_acc)

    return final_accuracy


def main(args):
    print("CUDA status: ", args.cuda)
    torch.cuda.manual_seed_all(args.seed)

    print(">>>> Preparing data...")
    train_files = glob.glob(args.data_folder + args.train_folder + "*.json")
    val_files = glob.glob(args.data_folder + args.val_folder + "*.json")

    train_loader = build_loaders(
        train_files, mode="train", batch_size=args.train_batch_size, args=args)
    # train_val_loader = build_loaders(
    #     train_files, mode="train", batch_size=args.val_batch_size, args=args)
    val_loader = build_loaders(
        val_files, mode="test", batch_size=args.val_batch_size, args=args)

    # Print data information
    print("Train files: ", len(train_files))
    print("Val files: ", len(val_files))

    print(">>>> Preparing model...")
    model = PrescriptionPill(args).cuda()

    print(">>>> Preparing optimizer...")
    matching_criterion = ContrastiveLoss()
    class_weights = torch.FloatTensor(CFG.labels_weight).cuda()
    graph_criterion = torch.nn.NLLLoss(weight=class_weights)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_accuracy = 0
    patience = 5
    epochs_no_improve = 0
    print(">>>> Training...")
    history = {"train_loss": [], "val_acc": []}
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, matching_criterion, graph_criterion, epoch)
        print(">>>> Train Validation...")
        train_val_acc = 0 #val(model, train_val_loader)
        print("Train accuracy: ", train_val_acc)

        print(">>>> Test Validation...")
        val_acc = val(model, val_loader)
        print("Val accuracy: ", val_acc)
        
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            epochs_no_improve = 0
            import os
            model_dir = "./logs/weights/"
            os.makedirs(model_dir, exist_ok=True)
            if getattr(args, 'use_ocr', False):
                model_name = "model_best_ocr.pth"
            else:
                model_name = "model_best_no_ocr.pth"
            model_path = model_dir + model_name
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            print(f"EarlyStopping counter: {epochs_no_improve} out of {patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

        wandb.log({"train_loss": train_loss,
                  "train_acc": train_val_acc, "val_acc": val_acc})

    # Plot learning curves
    import matplotlib.pyplot as plt
    actual_epochs = len(history["train_loss"])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs + 1), history["train_loss"], label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs + 1), history["val_acc"], label='Test Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Learning curves saved to learning_curves.png")

if __name__ == '__main__':
    parse_args = option()

    wandb.init(project="VAIPE-Pills-Prescription-Matching", name=parse_args.run_name, mode="disabled",
               config={
                   "train_batch_size": parse_args.train_batch_size,
                   "val_batch_size": parse_args.val_batch_size,
                   "epochs": parse_args.epochs,
                   "lr": parse_args.lr,
                   "seed": parse_args.seed
               })

    args = wandb.config
    wandb.define_metric("val_acc", summary="max")
    main(parse_args)
    wandb.finish()
