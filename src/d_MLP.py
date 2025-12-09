#!/usr/bin/env python3
"""
MLP classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
import argparse
from pathlib import Path


class UCF101Dataset(Dataset):
    """Dataset HDF5 pour UCF-101"""
    
    def __init__(self, h5_file, split='trainingSet'):
        self.h5_file = h5_file
        self.split = split
        self.video_ids = []
        self.labels = []
        
        with h5py.File(h5_file, 'r') as f:
            all_classes = set()
            for video_key in f.keys():
                if f[video_key].attrs['split'] == split:
                    all_classes.add(f[video_key].attrs['class'])
            
            self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
            
            for video_key in f.keys():
                if f[video_key].attrs['split'] == split:
                    self.video_ids.append(video_key)
                    class_name = f[video_key].attrs['class']
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            features = f[self.video_ids[idx]]['features'][:]  # (4, 2048)
            features = features.flatten()  # (8192,)
        
        return torch.FloatTensor(features), self.labels[idx]


class VideoMLP(nn.Module):
    """MLP pour classification"""
    
    def __init__(self, input_dim=8192, num_classes=101):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, loader, criterion, optimizer, device):
    """Entraîne une epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in tqdm(loader, desc="Train"):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Valide le modèle"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Val"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', default='./data/featuresResNet50.h5')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_dir', default='./models')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Datasets
    train_dataset = UCF101Dataset(args.features_file, 'trainingSet')
    val_dataset = UCF101Dataset(args.features_file, 'valSet')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)} vidéos")
    print(f"Val: {len(val_dataset)} vidéos")
    print(f"Classes: {len(train_dataset.class_to_idx)}\n")
    
    # Modèle
    model = VideoMLP(num_classes=len(train_dataset.class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Entraînement
    Path(args.output_dir).mkdir(exist_ok=True)
    best_acc = 0
    
    print("="*60)
    print("ENTRAÎNEMENT")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
            print(f"✅ Sauvegardé (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n{'='*50}")
    print(f"Meilleure Val Acc: {best_acc:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()