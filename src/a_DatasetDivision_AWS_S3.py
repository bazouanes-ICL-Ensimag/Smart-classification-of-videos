#!/usr/bin/env python3
"""
Ce programme a pour but de diviser le dataset UCF-101 en train/val/test selon les groupes
Train: g01-g18 (18 groupes)
Val: g19-g22 (4 groupes)
Test: g23-g25 (3 groupes)
"""

# Library AWS pour utiliser les services AWS depuis python. 
import boto3
import re
from pathlib import Path
from tqdm import tqdm


BUCKET_NAME = "ucf101-smart-classification-of-videos"

# Pour lister uniquement les fichiers de ce dossier. 
RAW_PREFIX = "raw/"

# On organise les trois dossiers (train/val/test) avec les numéros des groupes.
# Ces groupes sont équlibrés.
TRAIN_GROUPS = list(range(1, 19))  
VAL_GROUPS = list(range(19, 23))   
TEST_GROUPS = list(range(23, 26))  


def get_split_folder(group_num):
    """Retourne le dossier correspondant en se basant sur le numéro de groupe"""
    if group_num in TRAIN_GROUPS:
        return "trainingSet"
    elif group_num in VAL_GROUPS:
        return "valSet"
    elif group_num in TEST_GROUPS:
        return "testSet"
    else:
        raise ValueError(f"Numéro de groupe invalide: {group_num}. Il Doit être entre 1 et 25.")


def parse_video_filename(filename):
    """
    Parse le nom de fichier UCF-101: v_ClasseName_g01_c01.avi
    Retourne: (class_name, group_num, clip_num)
    """
    # .+? = un petit bloc de texte, le plus court possible.
    # \d+ = un nombre.
    pattern = r'v_(.+?)_g(\d+)_c(\d+)\.'
    match = re.search(pattern, filename)
    
    if match:
        class_name = match.group(1)
        group_num = int(match.group(2))
        clip_num = int(match.group(3))
        return class_name, group_num, clip_num
    raise ValueError(f"Filename ne correspond pas au pattern attendu: {filename}")


def list_videos_from_s3(s3_client):
    """Liste toutes les vidéos du bucket S3"""
    
    videos = []
    # Un paginator est un objet spécial qui va appeler l’API S3 plusieurs fois automatiquement
    # pour récupérer toutes les pages de données.
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=RAW_PREFIX):
        # Si la page ne contient pas des fichiers, on passe à la page suivante.
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            
            # Ignorer les dossiers S3
            if key.endswith('/'):
                continue
            
            
            if key.endswith('.avi'):
                videos.append(key)
    
    print(f"Dans note bucket, nous avons trouvé {len(videos)} vidéos.")
    return videos


def plan_reorganization(videos):
    """Planifie la réorganisation des vidéos"""
    print("Réorganisation en cours")
    
    copy_plan = []
    
    for source_key in videos:
        filename = Path(source_key).name
        class_name, group_num, clip_num = parse_video_filename(filename)
        
        split_folder = get_split_folder(group_num)
        
        destination_key = f"{split_folder}/g{group_num:02d}/{class_name}/{filename}"
        
        copy_plan.append({
            'source': source_key,
            'destination': destination_key,
            'split': split_folder,
            'group': group_num,
            'class': class_name
        })  
    return copy_plan  



def execute_reorganization(s3_client, copy_plan):
    """Copie les fichiers dans S3"""
    
    for item in tqdm(copy_plan):
        try:
            s3_client.copy_object(
                CopySource={'Bucket': BUCKET_NAME, 'Key': item['source']},
                Bucket=BUCKET_NAME,
                Key=item['destination']
            )
        except Exception as e:
            print(f"❌ Erreur: {item['source']}")


def main():
    print(" Réorganisation du dataset UCF-101 : ")
    
    s3_client = boto3.client('s3')
    
    videos = list_videos_from_s3(s3_client)
    
    if not videos:
        print("❌ Aucune vidéo trouvée!")
        return
    copy_plan = plan_reorganization(videos)
    execute_reorganization(s3_client, copy_plan)


if __name__ == "__main__":
    main()