#!/usr/bin/env python3
"""
Test rapide pour identifier les vidéos corrompues
"""

import cv2
from pathlib import Path
from tqdm import tqdm
import json
import argparse


def test_video(video_path):
    """Test si une vidéo peut s'ouvrir et lire 3 frames"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return False, "Cannot open"
        
        for i in range(4):
            ret, frame = cap.read()
            
            if not ret:
                cap.release()
                return False, f"Cannot read frame {i}"
            
            if frame is None:
                cap.release()
                return False, f"Frame {i} is None"
        
        cap.release()
        return True, "OK"
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Test vidéos corrompues')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output', default='./outputs/corrupted_videos.json')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    videos = sorted(data_dir.rglob("*.avi"))
    
    print(f"Test de {len(videos)} vidéos...")
    
    corrupted = []
    ok_count = 0
    
    for video in tqdm(videos, desc="Test vidéos"):
        is_ok, reason = test_video(video)
        
        if is_ok:
            ok_count += 1
        else:
            corrupted.append({
                'path': str(video),
                'name': video.name,
                'reason': reason
            })
    
    print(f"\n{'='*50}")
    print(f" Vidéos OK: {ok_count}/{len(videos)} ({ok_count/len(videos)*100:.1f}%)")
    print(f" Vidéos corrompues: {len(corrupted)}/{len(videos)} ({len(corrupted)/len(videos)*100:.1f}%)")
    print(f"{'='*50}")
    
    if corrupted:
        with open(args.output, 'w') as f:
            json.dump(corrupted, f, indent=2)
        print(f"\n Liste sauvegardée: {args.output}")
        
        print(f"\n Premières vidéos corrompues:")
        for vid in corrupted[:10]:
            print(f"  - {vid['name']}: {vid['reason']}")
    else:
        print(f"\n Aucune vidéo corrompue détectée!")


if __name__ == "__main__":
    main()
