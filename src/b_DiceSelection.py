#!/usr/bin/env python3
"""
Sélection adaptative des frames avec le coefficient de Dice
"""

import cv2
import numpy as np
from pathlib import Path



def compute_dice_coeff(frame1, frame2):
    """Calcule le taux de similarité entre 2 frames (0 à 1)"""
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    
    numerator = 2.0 * f1 * f2
    denominator = f1 * f1 + f2 * f2
    
    coeffs = np.divide(numerator, denominator, out=np.ones_like(numerator), where=(denominator != 0))
    
    return float(np.mean(coeffs))


def select_and_save_frames(video_path, output_dir):
    """Sélectionne et sauvegarde les frames d'une vidéo"""
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Impossible d'ouvrir : {video_path}")
    
    ret0, frame0 = cap.read()
    ret1, frame1 = cap.read()
    
    if not ret0 or not ret1:
        cap.release()
        raise ValueError(f"❌ Vidéo trop courte (moins de 2 frames) : {video_path}")
    
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    parts = video_path_obj.parts
    split = parts[-4]
    group = parts[-3]
    classe = parts[-2]
    
    
    out_dir = Path(output_dir) / split / group / classe
    out_dir.mkdir(parents=True, exist_ok=True)
    
    
    cv2.imwrite(str(out_dir / f"{video_name}_frame_0000.jpg"), frame0)
    cv2.imwrite(str(out_dir / f"{video_name}_frame_0001.jpg"), frame1)
    
    similarities = [compute_dice_coeff(frame0, frame1)]
    refer_frame = frame1
    frame_idx = 2
    n_total = 2
    n_selected = 2
    
    # Traiter frames suivantes
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        n_total += 1
        sim = compute_dice_coeff(refer_frame, frame)
        
        if sim < np.mean(similarities):
            cv2.imwrite(str(out_dir / f"{video_name}_frame_{frame_idx:04d}.jpg"), frame)
            refer_frame = frame
            similarities.append(sim)
            n_selected += 1
        
        frame_idx += 1
    
    cap.release()
    
    return {
        'video': video_name,
        'total': n_total,
        'selected': n_selected,
        'ratio': n_selected / n_total
    }


