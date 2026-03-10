# Fine-Grained Topic Classification with ALBERT — MuSeCarASTE Dataset

This repository contains the code, data setup, and results for **topic classification using new fine-grained triple-level labels**.

## Experiment Overview

- **Goal:** Compare topic classification performance using **fine-grained triple-level labels** vs. original MuSe segment-level labels.
- **Dataset:** MuSe dataset devel and train file with **subject-level split**:
  - Training: 136 videos  
  - Development: 45 videos  
  - Testing: 46 videos  
- **Metric:** F1 score (micro-average) and Unweighted Average Recall (UAR).  
- **Framework:** Hugging Face Transformers with **ALBERT-xxlarge-v2**.  
- **Training:** Distributed training on 4 GPUs using **PyTorch DDP**.

## Results

| Label Type | F1 Score test | Notes |
|------------|---------------|-------|
| Fine-grained triple-level labels | 82 | Current experiment |
| MuSe segment-level labels (original) | 70 | Baseline |


> **Note:** All experiments used identical segments for both labels for a fair comparison. Segments that do not yield any triple were dropped since fine-grained triple-level labels do not apply and are not aligned with them.

## Directory Structure

──

├── train136.csv                     # Training data (136 videos)

├── devel45.csv                      # Development data (45 videos)

├── test46.csv                        # Test data (46 videos, subset available)

├── topic_class_mapping_MuSe2020.csv # Mapping of topic labels

├── trainnewdd.py           # Main distributed training script

├── results_segment wise old labels/ # Model outputs, checkpoints, metrics, confusion matrices for old labels

├── results_triple wise labels/ # Model outputs, checkpoints, metrics, confusion matrices for new labels

├── requirements.txt #  Install Dependencies

├── environment.yml # Conda environment File

└── README.md                         # This file

## Usage

1. **Install dependencies:**

pip install -r requirements.txt

Or

conda env create -f environment.yml

2. **Run training:**

python scripts/train_ddp_albert.py --no_train False --no_eval_final False

> This launches distributed training on all available GPUs.

3. **Evaluation and metrics:**
   - F1, UAR, and TopicScore are automatically computed and saved.
   - Classification reports and confusion matrices are saved for each split (train/dev/test).
   - TensorBoard logs are saved in `logs/` for easy visualization.
.
## Notes

- Code uses **PyTorch DDP** for multi-GPU training. The code requires at least one GPU
- Fine-grained labels are at **triple-level granularity**; older labels are at **segment-level granularity** for baseline comparison, and can be accessed in original muse dataset and not included here.









