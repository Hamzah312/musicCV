# 🎧 Music Genre & Tag Classification using Deep Learning

This project explores **music understanding through deep learning** using **spectrogram-based EfficientNet models**.  
It includes **four experiments** across two datasets:

- **FMA-Small** → Single-label genre classification  
- **MagnaTagATune (MTAT)** → Single-label main-genre + multi-label tagging  

The work covers dataset preparation, spectrogram generation, 6-second chunk processing, EfficientNet training, augmentation, and evaluation.

---

## 📌 Project Overview

Music contains rich information:
- Genre  
- Instruments  
- Mood  
- Tempo  
- Vocals  

This project processes audio into **3-channel spectrogram images**:

| Channel | Description |
|--------|-------------|
| **Mel-spectrogram (256 bins)** | Timbre + frequency content |
| **MFCC** | Texture, articulation, brightness |
| **Chroma** | Harmony and pitch class |

Models are trained using:
- **EfficientNet-B2/B4**
- **Binary Cross Entropy (for multi-label)**
- **Cross Entropy (for single-label)**
- **FastAI**
- **SpecAugment** (time/frequency masking)
- **FP16 mixed precision**
- **Dynamic thresholding for multi-label**

---

# 🧪 Experiments

Below is a summary of the 4 experiments conducted.

---

## ✅ **Experiment 1 — FMA-Small (Baseline ResNet50)**

**Task:** Single-label classification (8 genres)  
**Dataset:** 8,000 tracks → ~75k spectrogram images  
**Model:** ResNet50  
**Goal:** Build baseline + verify preprocessing pipeline  

**Result:**  
**~59% accuracy**

---

## ✅ **Experiment 2 — FMA-Small (EfficientNet-B2/B4)**

**Task:** Same as Experiment 1, with enhanced audio features  
**Features Added:**
- 6-second chunks  
- 3-channel spectrograms (Mel + MFCC + Chroma)  
- SpecAugment  
- EfficientNet-B2 and B4  

**Result:**  
**~53% accuracy** (limited by dataset noise & size)

---

## ✅ **Experiment 3 — MTAT Single-Label ("Main Genre")**

Converted multi-label tags → a single dominant genre.

**Classes:**  
`classical, electronic, rock, ambient, folk, jazz, pop, hiphop, metal`

**Dataset Size After Cleaning:** ~45k spectrograms  
**Model:** EfficientNet-B4  

**Result:**  
**78% test accuracy**

---

## 🎯 **Experiment 4 — MTAT Multi-Label Tagging (Final Model)**

**Task:** Predict all relevant tags for each clip:  
- Instruments  
- Genre  
- Mood  
- Tempo  
- Vocals  

**Model:** EfficientNet-B2 (BCEWithLogitsLoss)  
**Evaluation:** Micro F1-score + threshold tuning  

**Result:**  
**F1 ≈ 0.43**  
Comparable to published MTAT research benchmarks.

---

# 📁 Project Structure
music-classification/
│
├── data/
│ ├── fma_small_tracks_genre_top.csv
│ ├── mtat_singlelabel_genres.csv
│ └── mtat_spectrogram_chunks.csv
│
├── notebooks/
│ ├── 03_MTAG_MainGenre.ipynb
│ ├── dataset_01_FMA_ResNet50.ipynb
│ ├── dataset_02_FMA_EfficientNet.ipynb
│ ├── dataset_04_MTAG_MultiLabel.ipynb
│ ├── Training_01_FMA_ResNet50.ipynb
│ ├── Training_02_FMA_EfficientNet.ipynb
│ └── Training_04_MTAG_MultiLabel.ipynb

│
├── models/
│ ├── effnet_b2_fma_small_v2_export.pkl
│ ├── effnet_b2_fma_small_v2.pth
│ ├── effnet_b2_MagnaTagATune_v2.pkl
│ ├── effnet_b2_MagnaTagATune_v2.pth
│ ├── effnet_b4_MagnaTagATune_v2_single_label.pkl
│ ├── effnet_b4_MagnaTagATune_v2_single_label.pth
│ ├── resnet50_fma_small_export.pkl
│ └── resnet50_fma_small_stage1.pth
│
├── README.md
└── requirements.txt

# 🙌 Acknowledgements

Datasets:  
- **FMA Dataset** (Defferrard et al.)  
- **MagnaTagATune (MTAT)**  
Tools:  
- FastAI  
- Librosa  
- PyTorch  
- OpenXLab (dataset hosting)