# =============================================
# Hugging Face ALBERT Classification — Multi-Column + Full DDP (4 GPUs)
# =============================================
"""
ALBERT baseline for category classification on [DATASET NAME].

- Supports multi-column input (aspect, opinion, review)
- Uses Distributed Data Parallel (DDP)
- Evaluates using F1 and UAR
- Includes unseen combination evaluation

Splits:
Train: 136 | Dev: 45 | Test: 46 (video-level split)

Usage:
python script.py --category base
python script.py --no_train --category main
"""
import os
import argparse
import warnings
import sys

# Suppress Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress PyTorch C++ backend logs
os.environ["PYTHONWARNINGS"] = "ignore"      # global Python warning filter
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress PyTorch C++ warnings
# Number of CPU threads per DDP process
threads_per_process = 4  # adjust depending on your CPU cores

os.environ["OMP_NUM_THREADS"] = str(threads_per_process)
os.environ["MKL_NUM_THREADS"] = str(threads_per_process)
from transformers import logging
logging.set_verbosity_error()

import gc
import torch
torch.backends.cudnn.benchmark = True

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import numpy as np
import pickle
import csv
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import GPUtil

# ---------------------------
# DDP utilities
# ---------------------------
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def free_gpu_cache(rank):
    if rank == 0:
        print("Initial GPU usage:")
        GPUtil.showUtilization()
    gc.collect()
    torch.cuda.empty_cache()
    if rank == 0:
        print("GPU usage after emptying cache:")
        GPUtil.showUtilization()

# ---------------------------
# Dataset class
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, terms, labels, tokenizer, secondary_cols=None, max_length=200):
        self.terms = terms
        self.labels = labels
        self.secondary_cols = secondary_cols if secondary_cols else {}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, idx):
        secondary = " [SEP] ".join([self.secondary_cols[col][idx] for col in self.secondary_cols])
        secondary = secondary if secondary else None
        encoding = self.tokenizer(
            self.terms[idx],
            text_pair=secondary,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ---------------------------
# Metrics
# ---------------------------
#def f1_metric(y_true, y_pred):
#    return round(f1_score(y_true, y_pred, average='micro')*100,3)

#def uar_metric(y_true, y_pred):
#    return round(recall_score(y_true, y_pred, average='macro')*100,3)
# ---------------------------
# Metrics
# ---------------------------
def f1_metric(y_true, y_pred, labels):
    """
    Micro F1 over all classes (transaction-level correctness).
    Includes zero-support classes via 'labels'.
    """
    return round(f1_score(y_true, y_pred, labels=labels, average='micro') * 100, 3)

def uar_metric(y_true, y_pred, labels):
    """
    Macro recall / UAR over all classes.
    Ensures rare/unseen classes are counted as zero if not predicted.
    """
    return round(recall_score(y_true, y_pred, labels=labels, average='macro') * 100, 3)
# ---------------------------
# Evaluation utilities
# ---------------------------
def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, title=None, ticklabels=None, path=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # pass labels here
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = ticklabels if ticklabels else [str(l) for l in labels]
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='Truth', xlabel='Prediction',
           title=title or 'Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], fmt), ha='center', va='center',
                    color='white' if cm[i,j]>thresh else 'black')
    fig.tight_layout()
    if path and title:
        fig.savefig(os.path.join(path, f"{title}.png"))
    return ax
def export_predictions(output_dir, y_true, y_pred, name):
    import tempfile
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    # Save y_true safely
    tmp_true = tempfile.NamedTemporaryFile(delete=False, dir=output_dir)
    with open(tmp_true.name, "wb") as f:
        pickle.dump(y_true, f, protocol=4)
    shutil.move(tmp_true.name, os.path.join(output_dir, f"{name}_y.pkl"))

    # Save y_pred safely
    tmp_pred = tempfile.NamedTemporaryFile(delete=False, dir=output_dir)
    with open(tmp_pred.name, "wb") as f:
        pickle.dump(y_pred, f, protocol=4)
    shutil.move(tmp_pred.name, os.path.join(output_dir, f"{name}_y_pred.pkl"))
from tqdm import tqdm

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    # wrap loader with tqdm, disable for non-rank-0 to avoid clutter
    pbar = tqdm(loader, desc=f"Rank {dist.get_rank()} Evaluation", disable=(dist.get_rank()!=0))

    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            preds = torch.argmax(outputs.logits, dim=1)
            labels = batch['labels']

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]

    dist.barrier()
    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_labels, all_labels)

    all_preds = torch.cat(gathered_preds).cpu().numpy()
    all_labels = torch.cat(gathered_labels).cpu().numpy()

    return all_labels, all_preds

def evaluate_loss(model, loader, device, loss_fct):
    model.eval()

    total_loss = torch.tensor(0.0).to(device)
    count = torch.tensor(0).to(device)

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = loss_fct(outputs.logits, batch['labels'])

            total_loss += loss * batch['labels'].size(0)
            count += batch['labels'].size(0)

    # combine results across GPUs
    dist.all_reduce(total_loss)
    dist.all_reduce(count)

    return (total_loss / count).item()

def classification_results(output_dir, y_true, y_pred, class_names, partition_name, writer=None,CATEGORY=None):
    os.makedirs(output_dir, exist_ok=True)
    labels = list(range(len(class_names)))

    f1_score_val = f1_metric(y_true, y_pred,labels=labels)
    uar_score_val = uar_metric(y_true, y_pred,labels=labels)
    category_score = 0.66*f1_score_val + 0.34*uar_score_val
    print(f"{partition_name} Metrics: F1={f1_score_val}, UAR={uar_score_val}, {CATEGORY} score={category_score}")


    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        digits=3,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_df['support'] = report_df['support'].fillna(0).astype(int)
    report_df.to_excel(os.path.join(output_dir, f"{partition_name}_classification_report.xlsx"))

    # Confusion matrix
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,            # ensure all classes included
        ticklabels=class_names,
        title=f"Confusion Matrix {partition_name}",
        path=output_dir
    )

    export_predictions(output_dir, y_true, y_pred, partition_name)

    # Save metrics CSV
    metrics_file = os.path.join(output_dir, f"{partition_name}_metrics.csv")
    with open(metrics_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['F1','UAR','category_score'])
        csv_writer.writerow([f1_score_val, uar_score_val, category_score])

    # TensorBoard
    if writer:
        writer.add_scalar(f'{partition_name}/F1', f1_score_val)
        writer.add_scalar(f'{partition_name}/UAR', uar_score_val)
        writer.add_scalar(f'{partition_name}/{CATEGORY} score', category_score)


def main_worker(rank, world_size, train_mode=True, FINAL_EVAL_MODE=True,CATEGORY=None):
    free_gpu_cache(rank)
    setup_distributed(rank, world_size)
    if rank == 0:
        print("Total GPUs available:", torch.cuda.device_count())
        print(f"Using {world_size} GPUs")
    device = torch.device(f'cuda:{rank}')
    print(f"Rank {rank} using GPU {torch.cuda.current_device()}")
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # ---------------------------
    # Parameters
    # ---------------------------
    Param_local = {
        'output_dir': f'./output_{CATEGORY}',
        'best_model_dir': f'./best_model_{CATEGORY}',
        'max_seq_length': 300,
        'train_batch_size': 5,
        'eval_batch_size': 12,
        'gradient_accumulation_steps': 1,
        'num_train_epochs': 5,
        'weight_decay': 0,
        'learning_rate': 1e-5,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'warmup_ratio': 0.06,
        'fp16': True,
        'manual_seed': 42,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 126,
        'save_steps': 400,
        'resume_from_checkpoint':None,
    }
    import random
    random.seed(Param_local['manual_seed'])
    np.random.seed(Param_local['manual_seed'])
    torch.cuda.manual_seed_all(Param_local['manual_seed'])


    # TensorBoard (only rank 0)
    writer = SummaryWriter(log_dir=f"./logs_{CATEGORY}") if rank==0 else None

    # ---------------------------
    # Load data dynamically based on CATEGORY
    # ---------------------------
    if CATEGORY == 'base':
        fields = ["aspect", "review","base_label"]
        secondary_cols_to_use = ['review']  # adjust if needed    

        def load_columns(df):
            terms = df['aspect'].tolist()
            labels = df['base_label'].tolist()
            secondary = {col: df[col].tolist() for col in secondary_cols_to_use or [] if col in df.columns}
            return terms, labels, secondary

        class_mapping_file = "museaste_base_class_mapping.csv"
        class_column = "base_category"
        unseen_cols = ['aspect']
        

    elif CATEGORY == 'main':
        fields = ["aspect", "base_primary", "MAIN_label"]
        secondary_cols_to_use = ['aspect']     

        def load_columns(df):
            terms = df['base_primary'].tolist()
            labels = df['MAIN_label'].tolist()
            secondary = {col: df[col].tolist() for col in secondary_cols_to_use or [] if col in df.columns}
            return terms, labels, secondary

        class_mapping_file = "museaste_MAIN_CATEGORY_mapping.csv"
        class_column = "MAIN_CATEGORY"
        unseen_cols = ['aspect']
        
    elif CATEGORY == 'topic':
        
        fields = ["review","aspect","opinion","label_topic"]
        secondary_cols_to_use = ["aspect","opinion"]
      

        def load_columns(df):
            terms = df['review'].tolist()
            labels = df['label_topic'].tolist()
            secondary = {col: df[col].tolist() for col in secondary_cols_to_use or [] if col in df.columns}
            return terms, labels, secondary

        class_mapping_file = "topic_class_mapping_MuSe2020.csv"
        class_column = "topic"
        unseen_cols = ['review','aspect', 'opinion']

    
    elif CATEGORY == 'topic_old_segment_labels':

        # ---------------------------
        # FOR OLD SEGMENT WISE LABELS
        # ---------------------------
        fields = ["review","label_topic"]
        secondary_cols_to_use = None
        

        def load_columns(df):
            terms = df['review'].tolist()
            labels = df['label_topic'].tolist()
            secondary = {col: df[col].tolist() for col in secondary_cols_to_use or [] if col in df.columns}
            return terms, labels, secondary

        class_mapping_file = "topic_class_mapping_MuSe2020.csv"
        class_column = "topic"
        unseen_cols = ['review']
    
    if CATEGORY != 'topic_old_segment_labels':
        train_df = pd.read_csv("train136.csv", usecols=fields, encoding="latin1")
        devel_df = pd.read_csv("devel45.csv", usecols=fields, encoding="latin1")
        test_file = "test46.csv"
    else:
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # FOR OLD SEGMENT WISE LABELS, use relevant topic labels from original Muse dataset, for the corresponding video and segment ids
        # ---------------------------------------------------------------------------------------------------------------------------------------
        train_df = pd.read_csv("train136oldlabels.csv", usecols=fields, encoding="latin1")
        devel_df = pd.read_csv("devel45oldlabels.csv", usecols=fields, encoding="latin1")
        test_file = "test46oldlabels.csv"    

    
    x_train, y_train, sec_train = load_columns(train_df)
    x_dev, y_dev, sec_dev = load_columns(devel_df)
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file, encoding="latin1", usecols=fields)
        x_test, y_test, sec_test = load_columns(test_df)
    else:
        x_test = y_test = sec_test = None

    num_labels = len(set(y_train))

    # ---------------------------
    # Model & tokenizer
    # ---------------------------
    model_name = "albert-xxlarge-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    # If not training, load the best model for evaluation
    if not train_mode:
        best_model_dir = Param_local['best_model_dir']
        if os.path.exists(best_model_dir):
            model = AutoModelForSequenceClassification.from_pretrained(
                best_model_dir,
                num_labels=num_labels
            )
            tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
            model.to(device)
            model = DDP(model, device_ids=[rank], output_device=rank)
            if rank == 0:
                print(f"Loaded best model from {best_model_dir}")# ---------------------------
    # Class weights & loss
    # ---------------------------
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fct = CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(enabled=Param_local['fp16'])

    # ---------------------------
    # Datasets & loaders
    # ---------------------------
    train_dataset = TextDataset(x_train, y_train, tokenizer, secondary_cols=sec_train, max_length=Param_local['max_seq_length'])
    dev_dataset   = TextDataset(x_dev, y_dev, tokenizer, secondary_cols=sec_dev, max_length=Param_local['max_seq_length'])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dev_sampler   = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Param_local['train_batch_size'],
        sampler=train_sampler,
        pin_memory=True,
        num_workers=threads_per_process
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=Param_local['eval_batch_size'],
        sampler=dev_sampler,
        pin_memory=True,
        num_workers=threads_per_process
    )

    if x_test is not None:
        test_dataset = TextDataset(x_test, y_test, tokenizer, secondary_cols=sec_test, max_length=Param_local['max_seq_length'])
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Param_local['eval_batch_size'],
            sampler=test_sampler,
            pin_memory=True,
            num_workers=threads_per_process
     )
        
    # ---------------------------
    # Optimizer & scheduler
    # ---------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=Param_local['learning_rate'],
        eps=Param_local['adam_epsilon'],
        weight_decay=Param_local['weight_decay']
    )

    num_training_steps = (
        Param_local['num_train_epochs']
        * len(train_loader)
        // Param_local['gradient_accumulation_steps']   
    )
    num_warmup_steps = int(num_training_steps * Param_local['warmup_ratio'])

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # ---------------------------
    # Load checkpoint if specified
    # ---------------------------
    if Param_local['resume_from_checkpoint']:
        checkpoint_dir = Param_local['resume_from_checkpoint']
        if rank==0:
            print(f"Resuming from checkpoint: {checkpoint_dir}")

        # ---------------------------
        # Load model weights
        # ---------------------------
        model_file_bin = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        model_file_safe = os.path.join(checkpoint_dir, 'model.safetensors')

        if os.path.exists(model_file_safe):
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safe)
            model.module.load_state_dict(state_dict)
            if rank==0:
                print("Loaded model from model.safetensors")
        elif os.path.exists(model_file_bin):
            model.module.load_state_dict(torch.load(model_file_bin, map_location=device))
            if rank==0:
                print("Loaded model from pytorch_model.bin")
        else:
            raise FileNotFoundError(f"No model file found in {checkpoint_dir}")

        # ---------------------------
        # Load optimizer state
        # ---------------------------
        optimizer_state_file = os.path.join(checkpoint_dir, 'optimizer.pt')
        if os.path.exists(optimizer_state_file):
            optimizer.load_state_dict(torch.load(optimizer_state_file, map_location=device))
            if rank==0:
                print("Loaded optimizer state")

        # ---------------------------
        # Load scheduler state
        # ---------------------------
        scheduler_state_file = os.path.join(checkpoint_dir, 'scheduler.pt')
        if os.path.exists(scheduler_state_file):
            scheduler.load_state_dict(torch.load(scheduler_state_file, map_location=device))
            if rank==0:
                print("Loaded scheduler state")

        # ---------------------------
        # Load global step
        # ---------------------------
        global_step_file = os.path.join(checkpoint_dir, 'global_step.pt')
        if os.path.exists(global_step_file):
            global_step = torch.load(global_step_file)
            if rank==0:
                print(f"Resuming training from global_step={global_step}")
        else:
            global_step = 0
            if rank==0:
                print("No global_step.pt found, starting from 0")

    else:
        global_step = 0
    # ---------------------------
    # Training loop (only if train_mode=True)
    # ---------------------------
    if train_mode:
        for epoch in range(Param_local['num_train_epochs']):
            model.train()
            train_sampler.set_epoch(epoch)
            pbar = tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch+1}", disable=(rank!=0))
            for step, batch in enumerate(pbar):
                batch = {k:v.to(device) for k,v in batch.items()}
                with autocast(enabled=Param_local['fp16']):
                    outputs = model(**batch)
                    loss = loss_fct(outputs.logits, batch['labels']) / Param_local['gradient_accumulation_steps']
                scaler.scale(loss).backward()

                if (step+1) % Param_local['gradient_accumulation_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Param_local['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    if rank == 0 and writer is not None:
                        pbar.set_postfix({"loss": loss.item()})
                        writer.add_scalar('train/loss', loss.item(), global_step)
                    del outputs, loss
                    

                    # Evaluation during training
                    if Param_local['evaluate_during_training'] and global_step % Param_local['evaluate_during_training_steps']==0:
                        class_names = pd.read_csv(class_mapping_file)[class_column].tolist()

                        dev_loss = evaluate_loss(model, dev_loader, device, loss_fct)

                        if rank == 0 and writer is not None:
                            writer.add_scalar('dev/loss', dev_loss, global_step)

                        y_true_dev, y_pred_dev = evaluate(model, dev_loader, device)

                        if rank == 0:
                            classification_results(
                            Param_local['output_dir'],
                            y_true_dev,
                            y_pred_dev,
                            class_names,
                            "dev",
                            writer,
                            CATEGORY
                         )
                        if rank == 0:
                            print(f"[Step {global_step}] Dev Loss: {dev_loss:.4f}")
                    # Save checkpoint
                    if rank==0 and global_step % Param_local['save_steps']==0:
                        checkpoint_dir = os.path.join(Param_local['output_dir'], f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        model.module.save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)

                        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))
                        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler.pt'))
                        torch.save(global_step, os.path.join(checkpoint_dir, 'global_step.pt'))

    if FINAL_EVAL_MODE:
        print(f"[Rank {rank}] Reached FINAL_EVAL_MODE block")
        class_names = pd.read_csv(class_mapping_file)[class_column].tolist()
        unseen_indices = []
        # Suppose CATEGORY columns for "unseen" check
        

        if x_test is not None:

            sec_train = sec_train or {}
            sec_test = sec_test or {}
            # Create unique (primary + secondary) tuples from training data
            train_tuples = set(
                tuple([x_train[i]] + [
                    sec_train[col][i] for col in unseen_cols if col in sec_train
                ])
                for i in range(len(x_train))
            )

            test_tuples = [
                tuple([x_test[i]] + [
                    sec_test[col][i] for col in unseen_cols if col in sec_test
                ])
                for i in range(len(x_test))
            ]

            # Identify test samples based on columns not seen in training
            unseen_indices = [i for i, t in enumerate(test_tuples) if t not in train_tuples]

    

        # Create loaders for all partitions
        partition_loaders = []
        partition_names = []

        # Train loader
        partition_loaders.append(train_loader)
        partition_names.append("train")

        # Dev loader
        partition_loaders.append(dev_loader)
        partition_names.append("dev")

        # Test loader
        if x_test is not None:
            partition_loaders.append(test_loader)
            partition_names.append("test")

        # Evaluate all partitions (all ranks participate)
        for loader, name in zip(partition_loaders, partition_names):
            y_true, y_pred = evaluate(model, loader, device)
            if rank == 0:
                classification_results(
                    Param_local['output_dir'],
                    y_true,
                    y_pred,
                    class_names,
                    name,
                    writer,
                    CATEGORY
                )
        if rank == 0:
            os.makedirs(Param_local['best_model_dir'], exist_ok=True)
            model.module.save_pretrained(Param_local['best_model_dir'])
            tokenizer.save_pretrained(Param_local['best_model_dir'])
            os.makedirs(Param_local['output_dir'], exist_ok=True)
            with open(os.path.join(Param_local['output_dir'], "config.txt"), "w") as f:
                for k, v in Param_local.items():
                    f.write(f"{k}: {v}\n")
            if writer:
                writer.close()
            
        # Evaluate unseen aspects separately (DDP-safe)
        if x_test and unseen_indices and CATEGORY in ['base', 'main']:
            unseen_terms = [x_test[i] for i in unseen_indices]
            unseen_labels = [y_test[i] for i in unseen_indices]
            unseen_sec = {col: [sec_test[col][i] for i in unseen_indices] for col in sec_test} if sec_test else None

            unseen_dataset = TextDataset(
                unseen_terms,
                unseen_labels,
                tokenizer,
                secondary_cols=unseen_sec,
                max_length=Param_local['max_seq_length']
            )

            unseen_sampler = DistributedSampler(
                unseen_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )

            unseen_loader = DataLoader(
                unseen_dataset,
                batch_size=Param_local['eval_batch_size'],
                sampler=unseen_sampler,
                pin_memory=True,
                num_workers=threads_per_process
            )

            y_true_unseen, y_pred_unseen = evaluate(model, unseen_loader, device)

            if rank == 0:
                print(f"Number of unseen {unseen_cols} in test set: {len(unseen_indices)}")
                classification_results(
                    Param_local['output_dir'],
                    y_true_unseen,
                    y_pred_unseen,
                    class_names,
                    "test_unseen",
                    writer,
                    CATEGORY
                )


    cleanup_distributed()

# ---------------------------
# Launch DDP with arguments
# ---------------------------
if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()

    # Train flag
    parser.add_argument('--no_train', action='store_false', dest='train', help='Skip training; by default training runs')

    # Final evaluation flag
    parser.add_argument('--no_eval_final', action='store_false', dest='eval_final', help='Skip final evaluation; by default final evaluation runs')

    # Category: optional, default None
    parser.add_argument('--category', type=str, default=None, choices=['topic','base', 'main', 'topic_old_segment_labels'],
                        help="Choose which category to train/evaluate: 'topic', 'base', 'main'. If not specified, all run sequentially.")

    args = parser.parse_args()

    TRAIN_MODE = args.train
    FINAL_EVAL_MODE = args.eval_final
    CATEGORY = args.category.lower() if args.category else None

    print(f"TRAIN_MODE={TRAIN_MODE}, FINAL_EVAL_MODE={FINAL_EVAL_MODE}, CATEGORY={CATEGORY}")

    # DDP spawn
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs detected!")
    if world_size == 1:
        print("Running in single-GPU mode")

    categories_to_run = [CATEGORY] if CATEGORY else ['topic', 'base', 'main', 'topic_old_segment_labels']

    for cat in categories_to_run:
        print(f"\n=== Running category: {cat} ===\n")
        torch.multiprocessing.spawn(
            main_worker,
            args=(world_size, TRAIN_MODE, FINAL_EVAL_MODE, cat),
            nprocs=world_size,
            join=True
        )   
