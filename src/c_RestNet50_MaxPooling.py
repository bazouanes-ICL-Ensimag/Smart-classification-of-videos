#!/usr/bin/env python3

"""
Extraction of features  ResNet50 + Max Pooling
"""
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np


# Extracteur ResNet50

class ResNet50FeatureExtractor:
    def __init__(self, device='cuda:0'):
        """  ResNet50 pré-entraîné sans la dernière couche """
        if not torch.cuda.is_available():
            raise RuntimeError("❌ GPU requis!")
        
        self.device = torch.device(device)

        resnet = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


    
    def extract_features(self, frames, batch_size=128):
        """Extrait features pour une liste de frames"""

        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i+batch_size]
                
                batch_tensors = []
                for frame in batch_frames:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(frame_rgb)
                    batch_tensors.append(tensor)
                
                batch_input = torch.stack(batch_tensors).to(self.device)
                features = self.model(batch_input)
                features = features.squeeze()
                
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                
                all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)


# Max Pooling

def temporal_max_pooling(features, num_segments=4):
    """
    Divise les features en 4 sous-groupes
    et prend le maximum de chaque segment
    
    Output: (4, 2048)
    """
    n_frames = features.shape[0]
    segment_size = n_frames / num_segments
    
    segment_features = []
    for i in range(num_segments):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        
        segment = features[start:end]
        segment_max = np.max(segment, axis=0)
        segment_features.append(segment_max)
    return np.stack(segment_features)  


# ========================================
# PIPELINE
# ========================================
# Pour chaque vidéo :
#   1. Extraction des features : ResNet50 → (nbre_frames, 2048)
#   2. Temporal Max Pooling → (4, 2048)
#   3. Sauvegarder
