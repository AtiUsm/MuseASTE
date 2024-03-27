# Generative ABSA

This repo contains the data and code of re-implementation of the paper [Towards Generative Aspect-Based Sentiment Analysis](https://aclanthology.org/2021.acl-short.64.pdf) in ACL 2021, under different experiment settings. The code repository by the original authors has been modified, with different data extraction method for muse, more detailed evaluation on ASTE task, and a different implementation of the main T5 model, to run aste task on our and semeval datasets. Furthermore, Code related to additional tasks have been removed and cleaned. But the repo follows the same technique as the original authors, hence the reference to the original authors is cited below.



## Quick Start

- Set up the environment as described in the above section. You can use gas.yml file to set up the environment or use the requirements file.
- Run command `sh run.sh`
- Use the code files in Code for Muse folder for Muse Dataset.
- Use the code files in Code for SemEval for SemEval Dataset.
- Put the combined muse train.csv and dev.csv files in data/aste/muse folder, by combining text data from muse dataset, and annotations from this repository.


## Detailed Usage
We conduct experiments on four ABSA tasks with four datasets in the paper, you can change the parameters in `run.sh` to try them:
```
python main.py --task $task \
            --dataset $dataset \
            --model_name_or_path t5-base \
            --paradigm $paradigm \
            --n_gpu 0 \
            --do_train \
            --do_eval \
            --train_batch_size 12 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 12 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 
```
- `$task` refers to one of the task of Aspect Sentiment Triplet Extraction in [`aste`] 
- `$dataset` refers to one of the four datasets in [`laptop14`, `rest14`, `rest15`, `rest6`,`muse`]
- `$paradigm` refers to one of the extraction paradigms proposed in the model. 

More details can be found in the paper and the help info in the `main.py`.



## Citation

If the code is used in your research, please star our repo and cite our paper as follows:
```
@inproceedings{zhang-etal-2021-towards,
    title = "Towards Generative Aspect-Based Sentiment Analysis",
    author = "Zhang, Wenxuan  and
      Li, Xin  and
      Deng, Yang  and
      Bing, Lidong  and
      Lam, Wai",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-short.64",
    pages = "504--510",
}
```
