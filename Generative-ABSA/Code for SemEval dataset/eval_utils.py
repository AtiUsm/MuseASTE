# This file contains the evaluation functions
import pandas as pd




def extract_triple(seq, task='aste'):
    extractions = []
    if task in ['aste']:
        all_pt = seq.split('; ')
        for pt in all_pt:
            pt = pt[1:-1]
            try:
                a, b, c = pt.split(', ')
            except ValueError:
                a, b, c = '', '', ''
            extractions.append((a, b, c))            
    return extractions

def extract_aspect(seq, task='aste'):
    extractions = []
    if task in ['aste']:
        all_pt = seq.split('; ')
        for pt in all_pt:
            pt = pt[1:-1]
            try:
                a, b, c = pt.split(', ')
            except ValueError:
                a, b, c = '', '', ''
            extractions.append((a))            
    return extractions

def extract_opinion(seq, task='aste'):
    extractions = []
    if task in ['aste']:
        all_pt = seq.split('; ')
        for pt in all_pt:
            pt = pt[1:-1]
            try:
                a, b, c = pt.split(', ')
            except ValueError:
                a, b, c = '', '', ''
            extractions.append((b))            
    return extractions

def extract_aopair(seq, task='aste'):
    extractions = []
    if task in ['aste']:
        all_pt = seq.split('; ')
        for pt in all_pt:
            pt = pt[1:-1]
            try:
                a, b, c = pt.split(', ')
            except ValueError:
                a, b, c = '', '', ''
            extractions.append((a, b))            
    return extractions

def extract_aspair(seq, task='aste'):
    extractions = []
    if task in ['aste']:
        all_pt = seq.split('; ')
        for pt in all_pt:
            pt = pt[1:-1]
            try:
                a, b, c = pt.split(', ')
            except ValueError:
                a, b, c = '', '', ''
            extractions.append((a, c))            
    return extractions


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in set(gold_pt[i]):
            if t in set(pred_pt[i]):
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(gold_labels,pred_labels, sents, io_format, task):
  assert len(gold_labels) == len(pred_labels) 
  num_samples = len(gold_labels)
  raw_scores=[]
  triple_gold=[]
  triple_pred=[]
  for i in range(num_samples):
    triple_gold.append(extract_triple(gold_labels[i]))
    triple_pred.append(extract_triple(pred_labels[i]))
  raw_scores.append({'triple':compute_f1_scores(triple_gold,triple_pred)})

  as_pair_gold=[]
  as_pair_pred=[]
  for i in range(num_samples):
    as_pair_gold.append(extract_aspair(gold_labels[i]))
    as_pair_pred.append(extract_aspair(pred_labels[i]))
  raw_scores.append({'aspect_sentiment_pair':compute_f1_scores(as_pair_pred,as_pair_gold)})


  ao_pair_gold=[]
  ao_pair_pred=[]
  for i in range(num_samples):
    ao_pair_gold.append(extract_aopair(gold_labels[i]))
    ao_pair_pred.append(extract_aopair(pred_labels[i]))
  raw_scores.append({'aspect_opinion_pair':compute_f1_scores(ao_pair_pred,ao_pair_gold)})

  a_gold=[]
  a_pred=[]
  for i in range(num_samples):
    a_gold.append(extract_aspect(gold_labels[i]))
    a_pred.append(extract_aspect(pred_labels[i]))

  raw_scores.append({'aspect':compute_f1_scores(a_pred,a_gold)})

  o_gold=[]
  o_pred=[]
  for i in range(num_samples):
    o_gold.append(extract_opinion(gold_labels[i]))
    o_pred.append(extract_opinion(pred_labels[i]))

  raw_scores.append({'opinion':compute_f1_scores(o_pred,o_gold)})
  print("\nResults")
  print(raw_scores)
  return raw_scores
