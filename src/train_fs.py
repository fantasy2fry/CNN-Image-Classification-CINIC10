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

# ==========================================
# Number of validation batches used for the fast per-epoch loss estimate.
# Full valid set (~90k images, 704 batches) is too slow to run every epoch.
# We sample a small fixed subset for a quick proxy metric during training.
# Full k-NN evaluation is always performed at the end on the test set.
# ==========================================
FAST_EVAL_BATCHES = 20   # 20 batches x 128 = ~2560 images — fast proxy


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

            # Pairwise L2 distances: Shape [B, N_support]
            test_emb  = model(images)
            distances = torch.cdist(test_emb, train_embeddings, p=2)

            # Indices of the k nearest neighbors (smallest distances): Shape [B, k]
            _, knn_indices = distances.topk(k, dim=1, largest=False)

            # Majority vote: pick the most frequent label per test sample
            knn_labels   = train_labels[knn_indices]
            predictions, _ = torch.mode(knn_labels, dim=1)

            correct += (predictions == labels).sum().item()
            total   += labels.size(0)

    return correct / total


def get_negatives(view, labels):
    """
    For each sample i in the batch, selects a random sample j such that
    labels[j] != labels[i] to serve as the triplet negative.

    Vectorized: builds the full [B, B] label-difference mask once before
    the loop, avoiding repeated nonzero() calls per sample.

    Fallback: cyclic shift when all batch samples share the same class.
    """
    B = view.shape[0]

    # diff_mask[i, j] = True  iff  labels[i] != labels[j]  — Shape: [B, B]
    diff_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    negatives = torch.zeros_like(view)
    for i in range(B):
        diff_idx = diff_mask[i].nonzero(as_tuple=True)[0]
        if len(diff_idx) > 0:
            neg_idx = diff_idx[torch.randint(len(diff_idx), (1,))].item()
        else:
            neg_idx = (i + 1) % B   # fallback: single-class batch
        negatives[i] = view[neg_idx]

    return negatives


def train_contrastive_epoch(model, dataloader, criterion, optimizer, device):
    """
    One full training epoch using TripletLoss.
      - anchor  : view1 (first augmented view)
      - positive: view2 (second augmented view of the same image)
      - negative: in-batch sample with a different class label
    """
    model.train()
    running_loss = 0.0
    total = 0

    loop = tqdm(dataloader, leave=False, desc="Train Contrastive")
    for inputs, labels in loop:
        view1, view2 = inputs[0].to(device), inputs[1].to(device)
        labels = labels.to(device)

        negatives = get_negatives(view1, labels).to(device)

        optimizer.zero_grad()

        anchor_emb = model(view1)
        pos_emb    = model(view2)
        neg_emb    = model(negatives)

        loss = criterion(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * view1.size(0)
        total += view1.size(0)
        loop.set_postfix(loss=(running_loss / total))

    return running_loss / total, 0.0


def evaluate_contrastive_fast(model, dataloader, criterion, device, max_batches=FAST_EVAL_BATCHES):
    """
    Fast per-epoch validation proxy: evaluates triplet loss over a small fixed
    number of batches instead of the full validation set (~90k images / 704 batches).

    This is purely a training monitor — NOT used as the final accuracy metric.
    The definitive evaluation is always k-NN on the full test set at the end.

    Args:
        max_batches: number of batches to process (None = full dataloader, slow).
    """
    model.eval()
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(inputs, list):
                view1, view2 = inputs[0].to(device), inputs[1].to(device)
            else:
                view1 = inputs.to(device)
                view2 = view1   # dummy positive for standard loaders

            labels    = labels.to(device)
            negatives = get_negatives(view1, labels).to(device)

            anchor_emb = model(view1)
            pos_emb    = model(view2)
            neg_emb    = model(negatives)

            loss = criterion(anchor_emb, pos_emb, neg_emb)

            running_loss += loss.item() * view1.size(0)
            total += view1.size(0)

    return (running_loss / total if total > 0 else 0.0), 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Few-Shot / Contrastive on CINIC-10")
    parser.add_argument('--model', type=str, default='mobilenet',
                        choices=['cnn', 'mobilenet', 'vgg11', 'resnet34'])
    parser.add_argument('--freeze_features', action='store_true',
                        help='Freeze the backbone; only the projection head trains')
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--lr',           type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout',      type=float, default=0.5,
                        help='Dropout probability (VGG-11 and Custom CNN only)')
    parser.add_argument('--use_cutout',   action='store_true')
    parser.add_argument('--optimizer',    type=str,   default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Images per class — enables few-shot mode')
    parser.add_argument('--disable_crop',     action='store_true')
    parser.add_argument('--disable_flip',     action='store_true')
    parser.add_argument('--disable_rotation', action='store_true')
    parser.add_argument('--margin',           type=float, default=1.0,
                        help='Margin for TripletLoss')
    parser.add_argument('--knn_every',        type=int, default=KNN_EVAL_EVERY,
                        help='Run k-NN evaluation every N epochs (default: 5)')
    parser.add_argument('--fast_eval_batches', type=int, default=FAST_EVAL_BATCHES,
                        help='Val batches for fast per-epoch proxy loss (0 = full val set, slow)')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    project_root    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir        = os.path.join(project_root, "data")
    experiments_dir = os.path.join(project_root, "experiments")
    models_dir      = os.path.join(project_root, "models")
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    fast_eval_batches = args.fast_eval_batches if args.fast_eval_batches > 0 else None

    # -----------------------------------------------------------------------
    # Experiment filename
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
        print(f"[*] SKIP: '{filename}' already exists. Delete to re-run.")
        return

    # -----------------------------------------------------------------------
    # Reproducibility & device
    # -----------------------------------------------------------------------
    set_seed(42)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"[*] Running on device: {device}")

    # -----------------------------------------------------------------------
    # Dataloaders
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
        is_contrastive=True,        # enables TwoCropTransform on train_loader only
        pretrained=is_pretrained_model
    )

    # -----------------------------------------------------------------------
    # Backbone — replace final classifier with Identity to expose features
    # -----------------------------------------------------------------------
    print(f"[*] Initializing backbone: {args.model.upper()}")

    if args.model == 'vgg11':
        backbone = VGG11(num_classes=10, dropout_rate=args.dropout)
        backbone.classifier[6] = nn.Identity()
        in_features = 4096
    elif args.model == 'resnet34':
        backbone = FinetunedResNet34(num_classes=10, freeze_features=args.freeze_features)
        backbone.model.fc = nn.Identity()
        in_features = 512
    elif args.model == 'mobilenet':
        backbone = get_finetuned_mobilenet(num_classes=10, freeze_features=args.freeze_features)
        backbone.classifier[0] = nn.Identity()
        in_features = 1280
    elif args.model == 'cnn':
        backbone = CustomCNN(num_classes=10, dropout_rate=args.dropout)
        backbone.fc = nn.Identity()
        in_features = 512

    # backbone -> Linear(in_features, 128) -> L2 normalize
    model = ContrastiveModel(backbone=backbone, in_features=in_features, embedding_dim=128)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[*] Parameters — Total: {total:,} | Trainable: {trainable:,}")

    # -----------------------------------------------------------------------
    # Loss & Optimizer
    # -----------------------------------------------------------------------
    criterion  = TripletLoss(margin=args.margin)
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
        f"Margin: {args.margin} | k-NN every: {args.knn_every} epochs | "
        f"Fast eval batches: {fast_eval_batches if fast_eval_batches else 'ALL (slow)'}"
    )
    start_time = time.time()
    last_val_knn_acc = float('nan')

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, _ = train_contrastive_epoch(model, train_loader, criterion, optimizer, device)

        # Fast proxy: only FAST_EVAL_BATCHES batches — avoids scanning 90k val images each epoch
        val_loss, _ = evaluate_contrastive_fast(
            model, valid_loader, criterion, device, max_batches=fast_eval_batches
        )

        # Full k-NN on valid: runs every knn_every epochs and always on the last epoch
        is_last_epoch = (epoch + 1 == args.epochs)
        run_knn = ((epoch + 1) % args.knn_every == 0) or is_last_epoch

        if run_knn:
            val_knn_acc      = get_knn_accuracy(model, train_loader, valid_loader, device, k=5)
            last_val_knn_acc = val_knn_acc
        else:
            val_knn_acc = last_val_knn_acc   # carry forward last known value

        epoch_time = time.time() - epoch_start
        knn_marker = " [k-NN]" if run_knn else ""
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss (fast): {val_loss:.4f} | "
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
    # Final definitive evaluation: 5-NN on full TEST set
    # -----------------------------------------------------------------------
    print(f"\n[*] Final evaluation: 5-NN on TEST set...")
    final_knn_acc = get_knn_accuracy(model, train_loader, test_loader, device, k=5)
    print(f"[*] FINAL k-NN TEST ACCURACY: {final_knn_acc:.4f}\n")

    results[-1]['Accuracy'] = final_knn_acc

    # -----------------------------------------------------------------------
    # Save CSV and model weights
    # -----------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"[*] Results saved to: {csv_path}")

    model_path = os.path.join(models_dir, filename.replace('.csv', '.pth'))
    torch.save(model.state_dict(), model_path)
    print(f"[*] Model weights saved to: {model_path}")


if __name__ == '__main__':
    main()