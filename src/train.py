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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, leave=False, desc="Training")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=(running_loss / total), acc=(correct / total))

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, leave=False, desc="Evaluating")
    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=(running_loss / total), acc=(correct / total))

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train Models on CINIC-10")
    # Unified model choices
    parser.add_argument('--model', type=str, default='vgg11',
                        choices=['cnn', 'mobilenet', 'vgg11', 'resnet34'],
                        help='Model architecture to train')
    parser.add_argument('--freeze_features', action='store_true',
                        help='Freeze feature extractor (Applies to ResNet34 & MobileNet)')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 Regularization (Weight Decay)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (Applies to VGG-11 & Custom CNN)')
    parser.add_argument('--use_cutout', action='store_true', help='Enable Cutout Data Augmentation')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Exact number of images per class for Few-Shot Learning')
    parser.add_argument('--disable_crop', action='store_true',
                        help='Disable Random Crop augmentation')
    parser.add_argument('--disable_flip', action='store_true',
                        help='Disable Random Horizontal Flip augmentation')
    parser.add_argument('--disable_rotation', action='store_true',
                        help='Disable Random Rotation augmentation')
    
    args = parser.parse_args()

    # Setup Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    experiments_dir = os.path.join(project_root, "experiments")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Dynamic Filename Generation
    filename = f"{args.model.upper()}_{args.optimizer}_{args.epochs}_E_{args.lr}_LR_"

    # Add dropout to filename only if the model uses it
    if args.model in ['vgg11', 'cnn']:
        filename += f"{args.dropout}_Drop_"

    # Add frozen tag if applicable
    if args.model in ['resnet34', 'mobilenet'] and args.freeze_features:
        filename += "Frozen_"

    if args.weight_decay > 0:
        filename += f"{args.weight_decay}_L2_"
    if args.use_cutout:
        filename += "Cutout_"

    # --- Few-Shot and Augmentation Tracking ---
    if args.samples_per_class is not None:
        filename += f"{args.samples_per_class}shots_"

    if args.disable_crop:
        filename += "NoCrop_"
    if args.disable_flip:
        filename += "NoFlip_"
    if args.disable_rotation:
        filename += "NoRot_"
    # -----------------------------------------------
    filename += ".csv"

    csv_path = os.path.join(experiments_dir, filename)

    # Idempotency: Skip if already exists
    if os.path.exists(csv_path):
        print(f"[*] SKIP: Results for experiment '{filename}' already exist.")
        print("[*] Skipping training to save time.")
        return

    # 1. Reproducibility
    set_seed(42)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Proceeding on device: {device}")

    # Set pretrained flag for ImageNet models (224x224 input)
    is_pretrained_model = args.model in ['resnet34', 'mobilenet']

    # 2. Get DataLoaders
    train_loader, valid_loader, test_loader = get_cinic10_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        samples_per_class=args.samples_per_class,
        use_crop=not args.disable_crop,
        use_horizontal_flip=not args.disable_flip,
        use_rotation=not args.disable_rotation,
        use_cutout=args.use_cutout,
        pretrained=is_pretrained_model
    )

    # 3. Initialize Model dynamically
    print(f"[*] Initializing {args.model.upper()}...")
    if args.model == 'vgg11':
        model = VGG11(num_classes=10, dropout_rate=args.dropout)
    elif args.model == 'cnn':
        model = CustomCNN(num_classes=10, dropout=args.dropout)
    elif args.model == 'resnet34':
        model = FinetunedResNet34(num_classes=10, freeze_features=args.freeze_features)
    elif args.model == 'mobilenet':
        model = get_finetuned_mobilenet(num_classes=10, freeze_features=args.freeze_features)

    model = model.to(device)

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    # Crucial for Transfer Learning: only pass trainable parameters to the optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # CSV Data Collection
    results = []

    # 5. Training Loop
    print(
        f"[*] Starting training: {args.model.upper()} | Opt: {args.optimizer.upper()} | Epochs: {args.epochs} | LR: {args.lr}")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        results.append({
            'Epoch': epoch + 1,
            'Train_Loss': train_loss, 'Train_Accuracy': train_acc,
            'Loss': val_loss, 'Accuracy': val_acc
        })

    total_time = time.time() - start_time
    print(f"[*] Training finished in {(total_time / 60):.2f} minutes.")

    # +++ FINAL TEST EVALUATION +++
    print(f"\n[*] Evaluating final model on completely unseen TEST set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[*] FINAL TEST LOSS: {test_loss:.4f} | FINAL TEST ACCURACY: {test_acc:.4f}\n")

    # 6. Save results to CSV
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