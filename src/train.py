import os
import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import get_cinic10_dataloaders, set_seed
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
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=(running_loss/total), acc=(correct/total))
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient tracking for evaluation to save memory
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
            
            loop.set_postfix(loss=(running_loss/total), acc=(correct/total))
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train CNN models on CINIC-10")
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mobilenet'], help='Model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 Regularization (Weight Decay)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability (only for Custom CNN currently)')
    parser.add_argument('--use_cutout', action='store_true', help='Enable Cutout Data Augmentation')
    
    args = parser.parse_args()
    
    # Setup Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    experiments_dir = os.path.join(project_root, "experiments")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Reproducibility
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Proceeding on device: {device}")
    
    # 2. Get DataLoaders
    # Note: MobileNet uses pretrained=True for 224x224 imgs and ImageNet normalization
    is_mobilenet = (args.model == 'mobilenet')
    train_loader, valid_loader, test_loader = get_cinic10_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=0,  # 0 to avoid Windows multiprocessing deadlock
        use_cutout=args.use_cutout,
        pretrained=is_mobilenet
    )
    
    # 3. Initialize Model
    if args.model == 'cnn':
        model = CustomCNN(num_classes=10)
        # Apply dropout to CustomCNN if requested in arguments
        if args.dropout > 0.0:
            model.dropout = nn.Dropout(p=args.dropout)
            # Re-wire forward to include dropout for experiments:
            # We override the forward method dynamically just for the experiment
            old_forward = model.forward
            def new_forward(x):
                x = model.conv1(x); x = model.bn1(x); x = F.relu(x); x = model.pool1(x)
                x = model.conv2(x); x = model.bn2(x); x = F.relu(x); x = model.pool2(x)
                x = model.conv3(x); x = model.bn3(x); x = F.relu(x); x = model.pool3(x)
                x = x.view(x.size(0), -1)
                x = model.fc1(x); x = F.relu(x); x = model.dropout(x); x = model.fc2(x)
                return x
            model.forward = new_forward
            
    else:
        model = get_finetuned_mobilenet(num_classes=10, freeze_features=True)
        # Assuming we don't dynamically alter MobileNet dropout for now, keep it simple.

    model = model.to(device)
    
    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Adam is our default, utilizing the user's arguments
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # CSV Data Collection
    results = []
    
    # 5. Training Loop
    print(f"[*] Starting training: {args.model.upper()} | Epochs: {args.epochs} | LR: {args.lr} | WD: {args.weight_decay}")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
              
        results.append({
            'Epoch': epoch + 1,
            'Train_Loss': train_loss, 'Train_Accuracy': train_acc,
            'Loss': val_loss, 'Accuracy': val_acc # Naming format to match your colleague's plots.ipynb
        })
        
    total_time = time.time() - start_time
    print(f"[*] Training finished in {(total_time/60):.2f} minutes.")
    
    # +++ FINAL TEST EVALUATION +++
    print(f"\n[*] Evaluating final model on completely unseen TEST set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[*] FINAL TEST LOSS: {test_loss:.4f} | FINAL TEST ACCURACY: {test_acc:.4f}\n")
    
    # 6. Save results to CSV (Using format from plots.ipynb)
    # E.g.: CustomCNN_10_E_0.001_LR_0.5_Drop.csv
    filename = f"{args.model}_{args.epochs}_E_{args.lr}_LR_"
    if args.dropout > 0: filename += f"{args.dropout}_Drop_"
    if args.weight_decay > 0: filename += f"{args.weight_decay}_L2_"
    if args.use_cutout: filename += "Cutout_"
    filename += ".csv"
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(experiments_dir, filename), index=False)
    print(f"[*] Saved results to {os.path.join('experiments', filename)}")
    
    # 7. Save Model Weights
    model_filename = filename.replace('.csv', '.pth')
    model_path = os.path.join(models_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"[*] Saved model weights to {os.path.join('models', model_filename)}")

if __name__ == '__main__':
    main()
