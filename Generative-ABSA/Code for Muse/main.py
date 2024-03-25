import argparse
import os

import time
import pickle


import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from data_utils import ABSADataset
from data_utils import get_extraction_aste_targets
from eval_utils import compute_scores
from torch.utils.data import random_split
from transformers import(
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import gc
import os

#os.environ['CUDA_VISIBLE_DEVICES']='2, 3'


def free_gpu_cache():
    #print(torch.cuda.memory_summary())
    print("Initial GPU Usage")
    gpu_usage()
    #print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()
   #print(torch.cuda.memory_summary())
    #cuda.select_device(0)
    #cuda.close()
    #cuda.select_device(0)


    print("GPU Usage after emptying the cache")
    gpu_usage()





    gc.collect()
    torch.cuda.empty_cache()
free_gpu_cache()

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='aste', type=str, required=True,
                        help="The name of the task, selected from: [aste]")
    parser.add_argument("--dataset", default='rest14', type=str, required=True,
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16, muse]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--tokenizer_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='extraction', type=str, required=True,
                        help="The way to construct target sentence, selected from: [extraction]")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
   
    parser.add_argument("--do_eval", action='store_true', 
                        help="Whether to run direct eval on the dev/test set.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    task_dir = f"./outputs/{args.task}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    task_dataset_dir = f"{task_dir}/{args.dataset}"
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    output_dir = f"{task_dataset_dir}/{args.paradigm}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type=type_path, 
                       paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)


def evaluate(data_loader, MODEL, paradigm, task, sents, TOKENIZER):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu}')
    MODEL.to(device)
    
    MODEL.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = MODEL.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=128)

        dec = [TOKENIZER.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [TOKENIZER.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        outputs.extend(dec)
        targets.extend(target)
      
    


    raw_scores = compute_scores(outputs, targets, sents, paradigm, task)
    results = {'raw_scores': raw_scores}
    # pickle.dump(results, open(f"{args.output_dir}/results-{args.task}-{args.dataset}-{args.paradigm}.pickle", 'wb'))

    return raw_scores, fixed_scores

def generate(data_loader, MODEL, paradigm, task, TOKENIZER):
    """
    Generate predictions and gold labels
    """
    
    
    device = torch.device(f'cuda:{args.n_gpu}')
    MODEL.to(device)
    
    MODEL.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = MODEL.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=128)

        dec = [TOKENIZER.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [TOKENIZER.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        outputs.extend(dec)
        targets.extend(target)
      
    
    data={'gold outputs':targets, 'predictions':outputs}
    df=pd.DataFrame(data)
    df.to_csv('devpredictions.csv')
      
    return outputs,targets

# initialization
args = init_args()
print("\n", "="*30, f"NEW EXP: {args.task.upper()} on {args.dataset}", "="*30, "\n")

seed_everything(args.seed)

tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

# show one sample to check the sanity of the code and the expected output
print(f"Here is an example (from dev set) under `{args.paradigm}` paradigm:")
dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='dev', 
                      paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
data_sample = dataset[2]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

# training process
if args.do_train:
    print("\n****** Conduct Training ******")
    
   
    #MODEL = T5FineTuner(args)


    free_gpu_cache()
    DEVICE = "cuda:0"
    TOKENIZER = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    MODEL = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, return_dict=True)
    OPTIMIZER = AdamW(MODEL.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=0.0)
    
    train_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='train', 
                      paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
    train_loader=DataLoader(train_dataset, batch_size=args.eval_batch_size, num_workers=4)
    val_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='dev', 
                      paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
    val_loader=DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4)

    train_loss = 0
    val_loss = 0
    train_batch_count = 0
    val_batch_count = 0
    MODEL = MODEL.to(DEVICE)
    for epoch in range(args.num_train_epochs):
        MODEL.train()
        for batch in tqdm(train_loader, desc="Training batches"):
            input_ids = batch["source_ids"].to(DEVICE)
            attention_mask = batch["source_mask"].to(DEVICE)
            labels = batch["target_ids"].to(DEVICE)
            decoder_attention_mask = batch["target_mask"].to(DEVICE)

            outputs = MODEL(
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_attention_mask=decoder_attention_mask
                        )

            OPTIMIZER.zero_grad()
            outputs.loss.backward()
            OPTIMIZER.step()
            train_loss += outputs.loss.item()
            train_batch_count += 1

            #Evaluation
            MODEL.eval()
        for batch in tqdm(val_loader, desc="Validation batches"):
            input_ids = batch["source_ids"].to(DEVICE)
            attention_mask = batch["source_mask"].to(DEVICE)
            labels = batch["target_ids"].to(DEVICE)
            decoder_attention_mask = batch["target_mask"].to(DEVICE)

            outputs = MODEL(
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_attention_mask=decoder_attention_mask
                        )

            OPTIMIZER.zero_grad()
            outputs.loss.backward()
            OPTIMIZER.step()
            val_loss += outputs.loss.item()
            val_batch_count += 1

        print(f"{epoch+1}/{2} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss/val_batch_count}")
    MODEL.save_pretrained("t5qa_model") #or path
    TOKENIZER.save_pretrained("t5qa_tokenizer") #or path
    free_gpu_cache()



# evaluation process
if args.do_eval:
    print("\n****** Conduct Evaluating with the last state ******")

    # print("Reload the model")
    TOKENIZER = T5Tokenizer.from_pretrained("t5qa_tokenizer") #or path
    MODEL = T5ForConditionalGeneration.from_pretrained("t5qa_model") #or path


    sents, inputs,targets = get_extraction_aste_targets(f'data/{args.task}/{args.dataset}/dev.csv')

    test_dataset = ABSADataset(TOKENIZER, data_dir=args.dataset, data_type='dev', 
                    paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    raw_scores = evaluate(test_loader, MODEL, args.paradigm, args.task, sents, TOKENIZER)
    outputs,targets= generate(test_loader, MODEL, args.paradigm, args.task, TOKENIZER)

    # write to file

    results_log_dir = './results_log' 
    log_file_path = f"results_log/{args.task}-{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = f"{args.task} on {args.dataset} under {args.paradigm}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
    exp_results = f"Triple raw F1 = {raw_scores['triple']['f1']:.4f}"
    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"
    if not os.path.exists(results_log_dir):
        os.mkdir(results_log_dir)
    if not os.path.exists(log_file_path):
          with open(log_file_path, "w+") as f:
            f.write(log_str)
    else:    
        with open(log_file_path, "a+") as f:
            f.write(log_str)
