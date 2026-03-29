import os
import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import get_cinic10_dataloaders, set_seed
from src.models_implementation.prototypical_net import PrototypicalResNet34, PrototypicalLoss


def train_one_epoch_proto(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, leave=False, desc="Training (ProtoNet)")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Otrzymujemy "embeddingi" (np. wektor 512-wymiarowy), a nie 10 klas!
        embeddings = model(inputs)

        # PrototypicalLoss zwraca i stratę (loss), i odległości (logits)
        loss, logits = criterion(embeddings, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Logits tutaj to odwrócone dystanse. Największa wartość = najbliższy prototyp.
        _, predicted = logits.max(1)

        # Ponieważ nasza funkcja straty zmapowała oryginalne labele (np. klasa 3, 7, 8)
        # do indeksów wewnątrz batcha (0, 1, 2), musimy zrobić to samo, żeby policzyć Accuracy.
        unique_classes = torch.unique(labels)
        target_indices = torch.zeros_like(labels)
        for i, c in enumerate(unique_classes):
            target_indices[labels == c] = i

        total += labels.size(0)
        correct += predicted.eq(target_indices).sum().item()

        loop.set_postfix(loss=(running_loss / total), acc=(correct / total))

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def compute_prototypes(model, dataloader, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            embeddings = model(inputs)
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())

    all_embeddings = torch.cat(embeddings_list)
    all_labels = torch.cat(labels_list)
    unique_classes = torch.unique(all_labels)

    prototypes = []
    prototype_labels = []

    for c in unique_classes:
        class_mask = (all_labels == c)
        class_embeddings = all_embeddings[class_mask]
        prototypes.append(class_embeddings.mean(dim=0))
        prototype_labels.append(c)

    return torch.stack(prototypes).to(device), torch.tensor(prototype_labels).to(device)


def evaluate_proto(model, dataloader, prototypes, prototype_labels, device):
    model.eval()
    running_loss = 0.0  # Dodajemy zmienną do śledzenia straty
    correct = 0
    total = 0

    loop = tqdm(dataloader, leave=False, desc="Evaluating (ProtoNet)")
    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            embeddings = model(inputs)

            # Liczymy odległości (dystans euklidesowy do kwadratu)
            distances = torch.cdist(embeddings, prototypes, p=2.0) ** 2

            # ==========================================
            # NOWE: LICZENIE STRATY (LOSS)
            # ==========================================
            # 1. Musimy dopasować prawdziwe etykiety z batcha do indeksów naszych prototypów
            target_indices = torch.zeros_like(labels)
            for i, c in enumerate(prototype_labels):
                target_indices[labels == c] = i

            # 2. Zamieniamy dystanse na logity (wartości ujemne)
            logits = -distances

            # 3. Liczymy Cross Entropy!
            loss = F.cross_entropy(logits, target_indices)
            running_loss += loss.item() * inputs.size(0)
            # ==========================================

            # Liczenie Accuracy (to co było)
            _, min_dist_indices = distances.min(dim=1)
            predicted_classes = prototype_labels[min_dist_indices]

            total += labels.size(0)
            correct += predicted_classes.eq(labels).sum().item()

            loop.set_postfix(loss=(running_loss / total), acc=(correct / total))

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train Prototypical Networks on CINIC-10")
    # Tniemy wybór modeli, bo skupiamy się tu tylko na architekturze ProtoNet (opartej na ResNet)
    parser.add_argument('--freeze_features', action='store_true',
                        help='Freeze feature extractor (Transfer Learning mode)')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4 — fine-tuning pretrained ResNet)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 Regularization / Weight Decay (default: 1e-4)')
    parser.add_argument('--use_cutout', action='store_true', help='Enable Cutout Data Augmentation')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of the embedding space (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate in the projection head (default: 0.3)')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='ReduceLROnPlateau patience in epochs (default: 3)')

    parser.add_argument('--samples_per_class', type=int, default=10,
                        help='Exact number of images per class for Few-Shot Learning (Default: 10)')

    parser.add_argument('--disable_crop', action='store_true', help='Disable Random Crop')
    parser.add_argument('--disable_flip', action='store_true', help='Disable Horizontal Flip')
    parser.add_argument('--disable_rotation', action='store_true', help='Disable Random Rotation')

    args = parser.parse_args()

    # Setup Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    experiments_dir = os.path.join(project_root, "experiments")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Dynamic Filename Generation
    filename = f"PROTONET_{args.optimizer}_{args.epochs}_E_{args.lr}_LR_"

    if args.freeze_features:
        filename += "Frozen_"

    if args.weight_decay > 0:
        filename += f"{args.weight_decay}_L2_"

    filename += f"{args.batch_size}_BS_"
    filename += f"emb{args.embedding_dim}_drop{args.dropout}_"

    if args.use_cutout:
        filename += "Cutout_"

    filename += f"{args.samples_per_class}shots_"

    if args.disable_crop:
        filename += "NoCrop_"
    if args.disable_flip:
        filename += "NoFlip_"
    if args.disable_rotation:
        filename += "NoRot_"

    filename += ".csv"

    csv_path = os.path.join(experiments_dir, filename)

    if os.path.exists(csv_path):
        print(f"[*] SKIP: Results for experiment '{filename}' already exist.")
        print("[*] Skipping training to save time.")
        return

    # 1. Reproducibility
    set_seed(42)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Proceeding on device: {device}")

    # 2. Get DataLoaders (ProtoNet ResNet uses 224x224, so pretrained=True)
    train_loader, valid_loader, test_loader = get_cinic10_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        samples_per_class=args.samples_per_class,
        use_crop=not args.disable_crop,
        use_horizontal_flip=not args.disable_flip,
        use_rotation=not args.disable_rotation,
        use_cutout=args.use_cutout,
        pretrained=True
    )

    # 3. Initialize PROTOTYPICAL Model
    print(f"[*] Initializing PROTOTYPICAL RESNET-34...")
    model = PrototypicalResNet34(
        freeze_features=args.freeze_features,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    # 4. Loss and Optimizer
    criterion = PrototypicalLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # ReduceLROnPlateau: cuts LR when train_loss stops improving.
    # This is the primary tool against the "loss plateaus, val degrades" pattern.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience
    )

    results = []

    # 5. Training Loop
    print(f"[*] Starting ProtoNet training | Opt: {args.optimizer.upper()} | Epochs: {args.epochs} | LR: {args.lr}")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch_proto(model, train_loader, criterion, optimizer, device)

        # Step scheduler on training loss — fires every epoch so we don't need val_loss
        scheduler.step(train_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            prototypes, prototype_labels = compute_prototypes(model, train_loader, device)
            val_loss, val_acc = evaluate_proto(model, valid_loader, prototypes, prototype_labels, device)
        else:
            val_loss, val_acc = None, None

        epoch_time = time.time() - epoch_start
        if val_loss is not None:
            print(f"Epoch {epoch + 1:02d}/{args.epochs} | Time: {epoch_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch + 1:02d}/{args.epochs} | Time: {epoch_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val: (evaluated every 5 epochs)")

        results.append({
            'Epoch': epoch + 1,
            'Train_Loss': train_loss, 'Train_Accuracy': train_acc,
            'Loss': val_loss if val_loss is not None else '',
            'Accuracy': val_acc if val_acc is not None else ''
        })

    total_time = time.time() - start_time
    print(f"[*] Training finished in {(total_time / 60):.2f} minutes.")

    print("\n[*] Wyliczanie ostatecznych prototypów ze zbioru treningowego...")
    final_prototypes, final_prototype_labels = compute_prototypes(model, train_loader, device)

    print("[*] Evaluating final model on the TRAINING set...")
    _, final_train_acc = evaluate_proto(model, train_loader, final_prototypes, final_prototype_labels, device)
    print(f"[*] FINAL TRAIN ACCURACY: {final_train_acc:.4f}")

    print("[*] Evaluating final model on completely unseen TEST set...")
    _, final_test_acc = evaluate_proto(model, test_loader, final_prototypes, final_prototype_labels, device)
    print(f"[*] FINAL TEST ACCURACY:  {final_test_acc:.4f}\n")

    # 6. Save results
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"[*] Saved results to {os.path.join('experiments', filename)}")

    # 7. Save Model Weights
    model_filename = filename.replace('.csv', '.pth')
    model_path = os.path.join(models_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"[*] Saved model weights to {os.path.join('models', model_filename)}")

if __name__ == '__main__':
    main()