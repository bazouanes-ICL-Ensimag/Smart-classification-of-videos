# Smart Classification of Videos - UCF-101

Classification intelligente des vidéos d'actions humaines du dataset UCF-101 utilisant un pipeline d'extraction de features ResNet50 et classification MLP.

## Analyse du dataset : 

Le rapport Rapport_de_stage_SCV.pdf permet d'étudier l'équilibre entre les classes et de prédire comment la sur-représentation ou la sous représentation  d'une classe peut affecter le modèle de classification. Dans ce contexte, nous justifions l'utilisation des groupes g01,...,g25 qui sont bien équilibrés. 

## 📁 Structure des Données

Le dataset UCF-101 a été divisé et réorganisé selon la structure suivante :
```
data/
├── trainingSet/
│   ├── g01/ ... g18/    (18 groupes)
│       └── [101 classes]/
│           └── *.avi
├── valSet/
│   └── g19/ ... g22/    (4 groupes)
└── testSet/
    └── g23/ ... g25/    (3 groupes)
```
Pour stocker ces données, j'ai utilisé le service AWS S3. Cependant, plus tard, lors de l'exécution des codes, la transmission des données a échoué car ce serveur n’a pas d’accès Internet.

Pour cela j'ai du retélécharger ces données sur le serveur, et stocker les frames extraites et les features sur ce serveur. 

---

## ⚙️ Environnement

### Serveur
**Narval (Compute Canada)** - Nécessite soumission de jobs SLURM.

### Environnement virtuel virtualPyEnv (qui contient toutes les libraries nécessaires) à activer dans chaque job SLURM. (Il faut veiller à ce que la version python soit compatible avec la librairie pytorch).

### Limitations
- ResNet50 pré-téléchargé dans `~/.cache/torch/`
- Temps d'attente dans la file avant l'exécution du code.

---

## Pipeline de Classification

### Phase 1 : Sélection Adaptative des Frames

**Objectif :** Réduire la redondance en sélectionnant les frames pertinentes et informatives

**Méthode :** Coefficient de Dice (similarité inter-frames)
- Conserve frames 0 et 1.
- Pour chaque frame suivante : si similarité < moyenne historique → garde la frame.


**Code :** `src/b_DiceSelection.py`

---

### Phase 2 : Extraction Features ResNet50 + Max Pooling Temporel

**Architecture :**
```
Frames JPG (N frames par vidéo)
    ↓
Chargement en RAM (OpenCV BGR)
    ↓
Conversion BGR → RGB
    ↓
Transformations ImageNet (resize 256, crop 224, normalisation)
    ↓
Batch de 128 frames : (128, 3, 224, 224) → GPU
    ↓
ResNet50 pré-entraîné (sans couche classification)
    ↓
Features : (N_frames, 2048)
    ↓
Max Pooling Temporel (4 segments)
    ↓
4 vecteurs : (4, 2048) par vidéo
    ↓
Sauvegarde HDF5 (featuresResNet50.h5)
```

**Code :** `src/extraction_features.py`

**Détails techniques :**
- 1 chargement ResNet50 par GPU (4 total)
- Batching GPU : 128 frames simultanées
- Max pooling par segments temporels (capture structure temporelle)

---

### Phase 3 : Entraînement MLP

**Architecture :**
```
Input : (8192,)  [4 × 2048 aplati]
    ↓
Dense(2048) + ReLU + Dropout(0.5)
    ↓
Dense(1024) + ReLU + Dropout(0.5)
    ↓
Dense(512) + ReLU + Dropout(0.3)
    ↓
Dense(101)  [softmax implicite dans CrossEntropyLoss]
```

**Code :** `src/d_MLP.py`

---

## 🔧 Dépendances
```bash
pip install torch torchvision opencv-python h5py numpy tqdm
```

**Modules Compute Canada :**
```bash
module load gcc opencv
```

---


