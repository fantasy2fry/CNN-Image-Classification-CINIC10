import os
import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import get_cinic10_dataloaders, set_seed
from src.models_implementation.vgg import VGG11
from src.models_implementation.resnet import FinetunedResNet34
from src.models_implementation.custom_cnn import CustomCNN
from src.models_implementation.mobilenet import get_finetuned_mobilenet

# Import Contrastive Models and Loss
from src.models_implementation.fs_contrastive import ContrastiveModel, TripletLoss


# ==========================================
# How often (in epochs) to run the expensive k-NN evaluation during training.
# k-NN requires a full pass over the train set to build the support index,
# so running it every epoch is too costly. We evaluate periodically and always
# run a final evaluation on the test set at the very end.
# ==========================================
KNN_EVAL_EVERY = 5


def get_knn_accuracy(model, train_loader, test_loader, device, k=5):
    """
    Evaluates the Contrastive Model using k-Nearest Neighbors (k-NN).

    Procedure:
      1. Forward-pass the entire support set (train_loader) to collect embeddings.
      2. For each test image, find the k closest support embeddings (L2 distance).
      3. Majority-vote over the k neighbors to predict the class label.

    Note: train_loader is in contrastive mode, so each batch is a list [view1, view2].
    We use only view1 as the canonical representative for the support index.
    """
    model.eval()

    train_embeddings = []
    train_labels = []

    print("[*] Extracting support embeddings for k-NN evaluation...")
    with torch.no_grad():
        for inputs, labels in train_loader:
            # In contrastive mode the dataloader returns [view1, view2]; take view1
            view1 = inputs[0].to(device) if isinstance(inputs, list) else inputs.to(device)
            labels = labels.to(device)

            emb = model(view1)
            train_embeddings.append(emb)
            train_labels.append(labels)

        # Shape: [N_support, embedding_dim]
        train_embeddings = torch.cat(train_embeddings, dim=0)
        # Shape: [N_support]
        train_labels = torch.cat(train_labels, dim=0)

        correct = 0
        total = 0

        for inputs, labels in test_loader:
            # test_loader always returns standard (images, labels) — no contrastive wrapping
            images = inputs.to(device)
            labels = labels.to(device)

            # Shape: [B, embedding_dim]
            test_emb = model(images)

            # Pairwise L2 distances between every test embedding and every support embedding
            # Shape: [B, N_support]
            distances = torch.cdist(test_emb, train_embeddings, p=2)

            # Indices of the k nearest neighbors (smallest distances)
            # Shape: [B, k]
            _, knn_indices = distances.topk(k, dim=1, largest=False)

            # Gather the labels of the k nearest neighbors
            knn_labels = train_labels[knn_indices]   # Shape: [B, k]

            # Majority vote: pick the most frequent label per test sample
            predictions, _ = torch.mode(knn_labels, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def get_negatives(view, labels):
    """
    For each sample i in the batch, selects a random sample j from the same batch
    such that labels[j] != labels[i]. This sample serves as the 'negative' in
    the Triplet Loss (anchor=view[i], positive=another view of i, negative=view[j]).

    Vectorized approach: we build the full [B, B] label-difference mask once and
    then sample from it, avoiding repeated Python-level calls to nonzero().

    Fallback: if the entire batch contains only one class, we use the cyclically
    shifted sample as a pseudo-negative (loss will be non-informative but training
    won't crash).
    """
    B = view.shape[0]

    # diff_mask[i, j] = True  iff  labels[i] != labels[j]
    # Shape: [B, B]
    diff_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    negatives = torch.zeros_like(view)

    for i in range(B):
        diff_idx = diff_mask[i].nonzero(as_tuple=True)[0]
        if len(diff_idx) > 0:
            rand_pick = torch.randint(len(diff_idx), (1,))
            neg_idx = diff_idx[rand_pick].item()
        else:
            # Fallback: cyclic shift — batch is single-class, no true negative available
            neg_idx = (i + 1) % B

        negatives[i] = view[neg_idx]

    return negatives


def train_contrastive_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one full training epoch for the contrastive (triplet) setup.

    Each batch yields two augmented views of the same images (view1, view2).
      - anchor  : view1
      - positive: view2  (same image, different augmentation -> should be close)
      - negative: a sample from the batch with a different class label (should be far)

    Returns:
        epoch_loss (float): average triplet loss over all samples.
        0.0: placeholder for accuracy (undefined in the triplet metric space).
    """
    model.train()
    running_loss = 0.0
    total = 0

    loop = tqdm(dataloader, leave=False, desc="Train Contrastive")
    for inputs, labels in loop:
        # inputs is [view1, view2] because is_contrastive=True in the dataloader
        view1, view2 = inputs[0].to(device), inputs[1].to(device)
        labels = labels.to(device)

        # Mine negatives from the current batch (in-batch negative mining)
        negatives = get_negatives(view1, labels).to(device)

        optimizer.zero_grad()

        # Forward pass: compute L2-normalized embeddings for all three roles
        anchor_emb   = model(view1)
        pos_emb      = model(view2)
        neg_emb      = model(negatives)

        # Triplet Loss: max(0, d(anchor, pos) - d(anchor, neg) + margin)
        loss = criterion(anchor_emb, pos_emb, neg_emb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * view1.size(0)
        total += view1.size(0)

        loop.set_postfix(loss=(running_loss / total))

    epoch_loss = running_loss / total
    # Accuracy is not directly defined for triplet training; return 0.0 as placeholder
    return epoch_loss, 0.0


def evaluate_contrastive(model, dataloader, criterion, device):
    """
    Evaluates the contrastive model on a validation set (triplet loss only).

    Handles both contrastive dataloaders (inputs = [view1, view2]) and
    standard dataloaders (inputs = tensor). For standard loaders the same
    image is used as both anchor and positive — the loss won't be meaningful,
    but lets us reuse this function for monitoring.

    Returns:
        epoch_loss (float): average triplet loss.
        0.0: placeholder for accuracy.
    """
    model.eval()
    running_loss = 0.0
    total = 0

    loop = tqdm(dataloader, leave=False, desc="Eval Contrastive")
    with torch.no_grad():
        for inputs, labels in loop:
            if isinstance(inputs, list):
                # Contrastive loader: two independently augmented views
                view1, view2 = inputs[0].to(device), inputs[1].to(device)
            else:
                # Standard loader: use the same image as a dummy positive
                view1 = inputs.to(device)
                view2 = view1

            labels = labels.to(device)

            negatives = get_negatives(view1, labels).to(device)

            anchor_emb = model(view1)
            pos_emb    = model(view2)
            neg_emb    = model(negatives)

            loss = criterion(anchor_emb, pos_emb, neg_emb)

            running_loss += loss.item() * view1.size(0)
            total += view1.size(0)

            loop.set_postfix(loss=(running_loss / total))

    epoch_loss = running_loss / total
    return epoch_loss, 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Few-Shot / Contrastive on CINIC-10")
    parser.add_argument('--model', type=str, default='mobilenet',
                        choices=['cnn', 'mobilenet', 'vgg11', 'resnet34'],
                        help='Model architecture to use as the backbone')
    parser.add_argument('--freeze_features', action='store_true',
                        help='Freeze the backbone feature extractor (only the projection head trains)')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 weight decay (regularization strength)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (applies to VGG-11 and Custom CNN)')
    parser.add_argument('--use_cutout', action='store_true',
                        help='Enable Cutout (RandomErasing) data augmentation')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer')

    # Few-Shot specific
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Number of images per class (enables few-shot mode)')

    # Augmentation ablation flags
    parser.add_argument('--disable_crop',     action='store_true', help='Disable RandomCrop augmentation')
    parser.add_argument('--disable_flip',     action='store_true', help='Disable RandomHorizontalFlip augmentation')
    parser.add_argument('--disable_rotation', action='store_true', help='Disable RandomRotation augmentation')

    # Triplet Loss margin
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for TripletLoss (how far negatives must be pushed)')

    # k-NN evaluation frequency
    parser.add_argument('--knn_every', type=int, default=KNN_EVAL_EVERY,
                        help='Run k-NN accuracy evaluation every N epochs (default: 5). '
                             'Set to 1 to evaluate every epoch (slow).')

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    project_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir       = os.path.join(project_root, "data")
    experiments_dir = os.path.join(project_root, "experiments")
    models_dir     = os.path.join(project_root, "models")
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Experiment filename — encodes all relevant hyperparameters
    # -----------------------------------------------------------------------
    filename = (
        f"CONTRASTIVE_{args.model.upper()}_{args.optimizer}_"
        f"{args.epochs}E_{args.lr}LR_M{args.margin}_"
    )
    if args.samples_per_class:
        filename += f"{args.samples_per_class}SHOT_"
    if args.model in ['resnet34', 'mobilenet'] and args.freeze_features:
        filename += "Frozen_"
    if args.weight_decay > 0:
        filename += f"WD{args.weight_decay}_"
    if args.disable_crop:
        filename += "NoCrop_"
    if args.disable_flip:
        filename += "NoFlip_"
    if args.disable_rotation:
        filename += "NoRot_"
    if args.use_cutout:
        filename += "Cutout_"
    filename += ".csv"

    csv_path = os.path.join(experiments_dir, filename)

    if os.path.exists(csv_path):
        print(f"[*] SKIP: Results for '{filename}' already exist. Delete the file to re-run.")
        return

    # -----------------------------------------------------------------------
    # Reproducibility & device
    # -----------------------------------------------------------------------
    set_seed(42)
    device = torch.device(
        'cuda'  if torch.cuda.is_available() else
        'mps'   if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"[*] Running on device: {device}")

    # -----------------------------------------------------------------------
    # Dataloaders
    # is_contrastive=True wraps train_loader with TwoCropTransform so that
    # each batch returns [view1, view2] — two differently augmented versions
    # of the same images. valid_loader and test_loader remain standard.
    # -----------------------------------------------------------------------
    is_pretrained_model = args.model in ['resnet34', 'mobilenet']

    train_loader, valid_loader, test_loader = get_cinic10_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        samples_per_class=args.samples_per_class,
        use_crop=not args.disable_crop,
        use_horizontal_flip=not args.disable_flip,
        use_rotation=not args.disable_rotation,
        use_cutout=args.use_cutout,
        is_contrastive=True,        # <-- critical: enables TwoCropTransform
        pretrained=is_pretrained_model
    )

    # -----------------------------------------------------------------------
    # Backbone initialization
    # The final classification head of each backbone is replaced with
    # nn.Identity() to expose raw feature vectors, which are then fed
    # into the ContrastiveModel's projection head.
    # -----------------------------------------------------------------------
    print(f"[*] Initializing backbone: {args.model.upper()}")

    if args.model == 'vgg11':
        backbone = VGG11(num_classes=10, dropout_rate=args.dropout)
        # Remove the 7-th (index 6) FC layer (the 10-class classifier)
        backbone.classifier[6] = nn.Identity()
        in_features = 4096

    elif args.model == 'resnet34':
        backbone = FinetunedResNet34(num_classes=10, freeze_features=args.freeze_features)
        backbone.model.fc = nn.Identity()
        in_features = 512

    elif args.model == 'mobilenet':
        backbone = get_finetuned_mobilenet(num_classes=10, freeze_features=args.freeze_features)
        # MobileNetV2 classifier is nn.Sequential([Dropout,] Linear).
        # We zero out only the linear part (index 0 when Dropout is commented out).
        backbone.classifier[0] = nn.Identity()
        in_features = 1280

    elif args.model == 'cnn':
        backbone = CustomCNN(num_classes=10, dropout_rate=args.dropout)
        # Assume the last layer is the classifier; replace it
        backbone.fc = nn.Identity()
        in_features = 512  # adjust if CustomCNN differs

    # Wrap the backbone in the ContrastiveModel:
    # backbone -> projection head (Linear 128-d) -> L2 normalize
    model = ContrastiveModel(backbone=backbone, in_features=in_features, embedding_dim=128)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # -----------------------------------------------------------------------
    # Loss & Optimizer
    # -----------------------------------------------------------------------
    criterion = TripletLoss(margin=args.margin)

    # Only pass parameters that require gradients (respects freeze_features)
    opt_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(opt_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    results = []
    print(
        f"[*] Starting Contrastive Training | Model: {args.model.upper()} | "
        f"Optimizer: {args.optimizer.upper()} | Epochs: {args.epochs} | "
        f"Margin: {args.margin} | k-NN every: {args.knn_every} epochs"
    )
    start_time = time.time()

    last_val_knn_acc = float('nan')   # cache the last known k-NN accuracy

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, _ = train_contrastive_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   _ = evaluate_contrastive(model, valid_loader, criterion, device)

        # ----------------------------------------------------------------
        # k-NN evaluation: run every `knn_every` epochs and on the last epoch.
        # This is the most expensive step — it requires a full forward pass
        # over the support set to build the embedding index.
        # ----------------------------------------------------------------
        is_last_epoch = (epoch + 1 == args.epochs)
        run_knn = ((epoch + 1) % args.knn_every == 0) or is_last_epoch

        if run_knn:
            val_knn_acc      = get_knn_accuracy(model, train_loader, valid_loader, device, k=5)
            last_val_knn_acc = val_knn_acc
        else:
            # Use the last computed value so the CSV doesn't have NaNs in early rows
            val_knn_acc = last_val_knn_acc

        epoch_time = time.time() - epoch_start
        knn_marker = " [k-NN]" if run_knn else ""
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val k-NN Acc: {val_knn_acc:.4f}{knn_marker}"
        )

        results.append({
            'Epoch':          epoch + 1,
            'Train_Loss':     train_loss,
            'Train_Accuracy': 0.0,          # undefined for triplet training
            'Loss':           val_loss,
            'Accuracy':       val_knn_acc,
            'KNN_Computed':   run_knn,
        })

    total_time = time.time() - start_time
    print(f"[*] Training finished in {total_time / 60:.2f} minutes.")

    # -----------------------------------------------------------------------
    # Final evaluation on TEST set using k-NN
    # This is the definitive metric: how well do the learned embeddings
    # generalise to unseen test images when classified by nearest neighbors
    # in the support (train) embedding space.
    # -----------------------------------------------------------------------
    print(f"\n[*] Final evaluation: 5-NN on TEST set...")
    final_knn_acc = get_knn_accuracy(model, train_loader, test_loader, device, k=5)
    print(f"[*] FINAL k-NN TEST ACCURACY: {final_knn_acc:.4f}\n")

    # Overwrite the last epoch's accuracy with the true test accuracy
    results[-1]['Accuracy'] = final_knn_acc

    # -----------------------------------------------------------------------
    # Save results CSV and model weights
    # -----------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"[*] Results saved to: {csv_path}")

    model_filename = filename.replace('.csv', '.pth')
    model_path = os.path.join(models_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"[*] Model weights saved to: {model_path}")


if __name__ == '__main__':
    main()