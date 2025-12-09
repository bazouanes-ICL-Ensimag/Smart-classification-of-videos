#!/usr/bin/env python3
"""
Extraction features ResNet50 + Max Pooling
"""

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import h5py
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import argparse
import time
import json


GPU_BATCH_SIZE = 128


class ResNet50FeatureExtractor:
    """Extracteur de features ResNet50"""
    
    def __init__(self, device='cuda:0'):
        if not torch.cuda.is_available():
            raise RuntimeError("❌ GPU requis!")
        
        self.device = torch.device(device)
        
        print(f"Chargement ResNet50 sur {device}...")
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


def temporal_max_pooling(features, num_segments=4):
    """
    Divise features en 4 segments temporels et fait max pooling
    
    Args:
        features: (N_frames, 2048)
        num_segments: 4
    
    Returns:
        (4, 2048)
    """
    n_frames = features.shape[0]
    segment_size = n_frames / num_segments
    segment_features = []
    
    for i in range(num_segments):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size) if i < num_segments - 1 else n_frames
        
        segment = features[start_idx:end_idx]
        segment_max = np.max(segment, axis=0)  # (2048,)
        segment_features.append(segment_max)
    
    return np.stack(segment_features)  # (4, 2048)


def list_frames_by_group(frames_dir):
    """Liste frames par groupe"""
    frames_path = Path(frames_dir)
    organized = {}
    
    for split_dir in frames_path.iterdir():
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        organized[split_name] = {}
        
        for group_dir in split_dir.iterdir():
            if not group_dir.is_dir():
                continue
            group_name = group_dir.name
            organized[split_name][group_name] = {}
            
            for class_dir in group_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                
                frames_by_video = {}
                for frame_file in sorted(class_dir.glob("*.jpg")):
                    video_name = '_'.join(frame_file.stem.split('_')[:-2])
                    if video_name not in frames_by_video:
                        frames_by_video[video_name] = []
                    frames_by_video[video_name].append(str(frame_file))
                
                organized[split_name][group_name][class_name] = frames_by_video
    
    return organized


def extract_video(video_name, frame_paths, extractor, split_name, group_name, class_name):
    """Extrait features pour une vidéo"""
    # Charger frames
    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
    
    if len(frames) == 0:
        raise ValueError(f"❌ Aucune frame pour {video_name}")
    
    # Extraire features
    features = extractor.extract_features(frames, batch_size=GPU_BATCH_SIZE)
    
    # Max pooling temporel
    video_features = temporal_max_pooling(features, num_segments=4)
    
    return {
        'split': split_name,
        'group': group_name,
        'class': class_name,
        'video_name': video_name,
        'features': video_features,
        'n_frames': len(frames)
    }


def process_worker(worker_id, tasks, frames_dir, num_workers):
    """Worker qui traite sa part de groupes"""
    
    gpu_id = worker_id % torch.cuda.device_count()
    device = f'cuda:{gpu_id}'
    
    print(f"Worker {worker_id} → GPU {gpu_id}")
    
    extractor = ResNet50FeatureExtractor(device=device)
    
    results = []
    
    for split_name, group_name, group_data in tasks:
        for class_name, videos in group_data.items():
            for video_name, frame_paths in videos.items():
                try:
                    result = extract_video(
                        video_name, frame_paths, extractor,
                        split_name, group_name, class_name
                    )
                    results.append(result)
                except Exception as e:
                    print(f"\n❌ Erreur {video_name}: {e}")
    
    return results


def save_to_hdf5(results, output_file):
    """Sauvegarde features dans HDF5"""
    
    with h5py.File(output_file, 'w') as hf:
        for idx, result in enumerate(tqdm(results, desc="Sauvegarde HDF5")):
            grp = hf.create_group(f'video_{idx:05d}')
            
            grp.create_dataset('features', data=result['features'],
                             compression='gzip', compression_opts=4)
            
            grp.attrs['split'] = result['split']
            grp.attrs['group'] = result['group']
            grp.attrs['class'] = result['class']
            grp.attrs['video_name'] = result['video_name']
            grp.attrs['n_frames'] = result['n_frames']
    
    print(f"✅ HDF5 créé: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', default='./data/DiceFrames')
    parser.add_argument('--output_file', default='./data/featuresResNet50')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--corrupted_list', default='./outputs/corrupted_videos.json')
    
    args = parser.parse_args()
    
    print("="*50)
    print("EXTRACTION + TEMPORAL MAX POOLING")
    print("="*50)
    print(f"GPUs: {torch.cuda.device_count()}")
    print(f"Workers: {args.num_workers}")
    
    # Lister frames
    print("\n Listing frames...")
    organized = list_frames_by_group(args.frames_dir)
    
    # Filtrer corrompues
    if Path(args.corrupted_list).exists():
        with open(args.corrupted_list, 'r') as f:
            corrupted = {v['name'].replace('.avi', '') for v in json.load(f)}
        print(f" {len(corrupted)} vidéos corrompues filtrées")
        
        for split in organized.values():
            for group in split.values():
                for class_name in group:
                    group[class_name] = {
                        v: f for v, f in group[class_name].items() 
                        if v not in corrupted
                    }
    
    
    tasks = []
    for split_name, groups in organized.items():
        for group_name, group_data in groups.items():
            tasks.append((split_name, group_name, group_data))
    
    print(f" {len(tasks)} groupes à traiter")
    
    
    tasks_per_worker = [[] for _ in range(args.num_workers)]
    for i, task in enumerate(tasks):
        tasks_per_worker[i % args.num_workers].append(task)
    
    
    print(f"\n Extraction...")
    all_results = []
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_worker, i, tasks_per_worker[i], args.frames_dir, args.num_workers)
            for i in range(args.num_workers)
        ]
        
        for future in tqdm(futures, desc="Workers"):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                raise RuntimeError(f"❌ Worker échoué: {e}")
    
    elapsed = time.time() - start
    
    # Sauvegarder
    print(f"\n Sauvegarde...")
    save_to_hdf5(all_results, args.output_file)
    
    # Stats
    print(f"\n{'='*50}")
    print(f"Terminé!")
    print(f"Vidéos: {len(all_results)}")
    print(f"Temps: {elapsed/60:.1f} min")
    print(f"Vitesse: {len(all_results)/elapsed:.1f} vid/s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()