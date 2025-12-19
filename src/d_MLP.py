#!/usr/bin/env python3
"""
MLP Classifier 
"""

import torch
import torch.nn as nn


class VideoMLP(nn.Module):
    """
    MLP avec BatchNorm pour classification de vidéos
    
    Input:  (8192,) = 4 segments × 2048 features ResNet50
    Output: (101,) = probabilités pour 101 classes d'actions
    """
    
    def __init__(self, input_dim=8192, num_classes=101):
        super().__init__()
        
        self.network = nn.Sequential(
            # Couche 1
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),     
            nn.ELU(inplace=True),       
            nn.Dropout(0.4),
            
            # Couche 2
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            
            # Couche finale
            nn.Linear(1024, num_classes)
        )
        
    
    def forward(self, x):
        return self.network(x)


