#!/usr/bin/env bash

python main.py --task aste \
            --dataset muse \
            --model_name_or_path t5-base \
            --paradigm extraction \
            --n_gpu 0 \
            --do_eval\
            --train_batch_size 12\
            --gradient_accumulation_steps 1 \
            --eval_batch_size 12 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 \
