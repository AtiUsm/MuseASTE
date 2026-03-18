# Fine-Grained Topic Classification, Base Category Classification and Main Category Classification with BASELINE ALBERT — MuSeCarASTE Dataset

This repository contains the code, data setup, and results for **topic classification using new fine-grained triple-level labels**, base category claafication, and main category classifcation.

## Experiment Overview

- **Goal:** Compare **topic** classification performance using **fine-grained triple-level labels** vs. original MuSe segment-level labels.
             Evaluate **Base and Main Category** Classification performance.
- **Dataset:** MuSe dataset devel and train file with **subject-level split**:
  - Training: 136 videos  
  - Development: 45 videos  
  - Testing: 46 videos  
- **Metric:** F1 score (micro-average) and Unweighted Average Recall (UAR).  
- **Framework:** Hugging Face Transformers with **ALBERT-xxlarge-v2**.  
- **Training:** Distributed training on 4 GPUs using **PyTorch DDP**.

## Results

1. Topic Category (triple and segment wise)

| Label Type | F1 Score test | Notes |
|------------|---------------|-------|
| Fine-grained triple-level labels | 82 | Current experiment |
| MuSe segment-level labels (original) | 70 | Baseline |

2. Base Category 

| Label Type | F1 Score test | Notes |
|------------|---------------|-------|
| Base Aspect labels | 95 | score on all test including all columns |
| Base Aspect unseen | 81 |  score on 792 useen aspects alone|


2. Main Category 

| Label Type | F1 Score test | Notes |
|------------|---------------|-------|
| Main Category | 99 | current experiment |

> **Note:** All experiments used identical segments for all labels for a fair comparison. Segments that do not yield any triple were dropped since fine-grained triple-level labels do not apply and are not aligned with them.

## Directory Structure
.

├── train136.csv                     # Training data (136 videos)

├── devel45.csv                      # Development data (45 videos)

├── test46.csv                        # Test data (46 videos)

├── datasetpartition.csv              # Video ids and the corresponding partition (train, test, devel) that they belong to.

├── topic_class_mapping_MuSe2020.csv # Mapping of topic labels

├── museaste_base_class_mapping.csv # Mapping of base category labels

├── museaste_MAIN_CATEGORY_mapping.csv # Mapping of main category labels

├── TrainCategory.py           # Main distributed training script

├── results_segment wise old labels/ # Model outputs, checkpoints, metrics, confusion matrices for old topic labels

├── results_triple wise labels/ # Model outputs, checkpoints, metrics, confusion matrices for new topic labels

├── results base category/ # Model outputs, checkpoints, metrics, confusion matrices for new base labels

├── results main category/ # Model outputs, checkpoints, metrics, confusion matrices for new main labels

├── requirements.txt #  Install Dependencies

├── environment.yml # Conda environment File

└── README.md                         # This file

## Usage

1. **Install dependencies:**

pip install -r requirements.txt

Or

conda env create -f environment.yml

2. **Run training & evaluation:**

python  TrainCategory.py --category main

python TrainCategory.py --no_train --category base

python TrainCategory.py --no_eval --category topic

> This launches distributed training on all available GPUs.


3. **Evaluation and metrics:**
   - F1, UAR, and TopicScore are automatically computed and saved.
   - Classification reports and confusion matrices are saved for each split (train/dev/test).
   - TensorBoard logs are saved in `logs/` for easy visualization.
.
## Notes

- Code uses **PyTorch DDP** for multi-GPU training. The code requires at least one GPU
- Fine-grained labels are at **triple-level granularity**; older labels are at **segment-level granularity** for baseline comparison, and can be accessed in original muse dataset and not included here.













