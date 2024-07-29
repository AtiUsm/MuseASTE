
    source="14lap"
      python3 ../../train.py --log_dir ./log/in-domain/$source \
        --mode test \
        --source ${source} \
        --target ${source} \
        --batch_size 8 \
        --save_model_dir ./save_models/${source}/ \
        #--model_name ${model_name}

     
