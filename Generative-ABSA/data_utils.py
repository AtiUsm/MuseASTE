# This file contains all data loading and transformation functions

import time
from torch.utils.data import Dataset
import pandas as pd

senttag2word = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}
def get_extraction_aste_targets(data_path):
    df= pd.read_csv(data_path)
    ids=df['id']
    ids=list(ids)
    ids=set(ids)
    label_dict=[]
    flag=0
    label=''
    context=[]
    for id in ids:
        sf=df.loc[df['id']==id]
        s_ids=sf['segment_id']
        s_id=list(s_ids)
        s_ids=set(s_id)
        for s_id in s_ids:
            nf=sf.loc[sf['segment_id']==s_id]
            flag=0
            for ind in nf.index:

                if flag==0:
                    text=nf['text'][ind]
                    curr_id=str(id)+'-'+str(s_id)
                    label=''
                    if nf["aspect"][ind]!="-":
                        label+='(' + nf["aspect"][ind] +', '+ nf["opinion"][ind]+', '+ senttag2word[nf["sentiment"][ind]]+')'
                    else:
                        label= "(None" +','+ "None" +','+ "None)"

                if flag==1:
                    if nf["aspect"][ind]!="-":
                        #print(nf["sentiment"][ind])
                        label=label+'; ('+nf["aspect"][ind] +', '+ nf["opinion"][ind]+', '+ senttag2word[nf["sentiment"][ind]]+')'
                flag=1
            label_dict.append({curr_id:label})
            context.append({curr_id:text})
    targets=[v  for i in label_dict  for k,v in i.items()]
    inputs=[c.split()  for i in context  for k,c in i.items()]
    sents=[c  for i in context  for k,c in i.items()]
    return sents, inputs, targets

def get_transformed_io(data_path, paradigm, task):
    """
    The main function to transform the Input & Output according to 
    the specified paradigm and task
    """
    #sents, labels = read_line_examples_from_file(data_path)

    # the input is just the raw sentence
    #inputs = [s.copy() for s in sents]

    # Get target according to the paradigm
    # annotate the sents (with label info) as targets
    # directly treat label infor as the target
    if paradigm == 'extraction':
        if task == 'aste':
            sents,inputs,targets = get_extraction_aste_targets(data_path)
        else:
            raise NotImplementedError
    else:
        print('Unsupported paradigm!')
        raise NotImplementedError 

    return sents, inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, paradigm, task, max_len=128):
        # 'data/aste/rest16/train.txt'
        self.data_path = f'data/{task}/{data_dir}/{data_type}.csv'
        self.paradigm = paradigm
        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()      # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        sents, inputs, targets = get_transformed_io(self.data_path, self.paradigm, self.task)

        for i in range(len(inputs)):

            input = ' '.join(inputs[i]) 
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding='max_length', truncation=True,
              return_tensors="pt",
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding='max_length', truncation=True,
              return_tensors="pt"
            )
            #print(len(tokenized_input)
            #print(len(tokenized_target)

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
