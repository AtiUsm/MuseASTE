source="16res"

    python main.py --log_dir log/in-domain/${source} \
    --tmp_log tmp_log/${source} \
    --source ${source} \
    --target ${source} \
    --mode train \
    --batch_size 2 \
    --model_dir save_models/${source} \


    python main.py --log_dir log/in-domain/${source} \
    --tmp_log tmp_log/${source} \
    --source ${source} \
    --target ${source} \
    --mode test \
    --batch_size 2 \
    --model_dir save_models/${source} \

