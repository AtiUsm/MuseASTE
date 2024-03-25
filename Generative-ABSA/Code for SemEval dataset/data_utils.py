# This file contains all data loading and transformation functions

import time
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}


def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    print(f"Total examples = {len(sents)}")
    return sents, labels







def get_extraction_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])
            c = senttag2word[tri[2]]
            all_tri.append((a, b, c))
        label_strs = ['('+', '.join(l)+')' for l in all_tri]
        targets.append('; '.join(label_strs))
    return targets


def get_transformed_io(data_path, paradigm, task):
    """
    The main function to transform the Input & Output according to 
    the specified paradigm and task
    """
    sents, labels = read_line_examples_from_file(data_path)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]
    # directly treat label infor as the target
    if paradigm == 'extraction':
        if task == 'aste':
            targets = get_extraction_aste_targets(sents, labels)
        else:
            raise NotImplementedError
    else:
        print('Unsupported paradigm!')
        raise NotImplementedError 

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, paradigm, task, max_len=128):
        # 'data/aste/rest16/train.txt'
        self.data_path = f'data/{task}/{data_dir}/{data_type}.txt'
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

        inputs, targets = get_transformed_io(self.data_path, self.paradigm, self.task)

        for i in range(len(inputs)):

            input = ' '.join(inputs[i]) 

            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt",
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt"
            )
            #print(len(tokenized_input)
            #print(len(tokenized_target)

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


