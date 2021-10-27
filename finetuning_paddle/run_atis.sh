python3 main.py --task atis \
                --model_type bert \
                --model_dir atis_model \
                --do_train --do_eval \
                --use_crf \
                --data_dir ../data \
                --model_name_or_path ../atis_params \
                --num_train_epochs 30.0