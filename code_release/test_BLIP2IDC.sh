#!/usr/bin/env bash

source /your_venv/bin/activate 

EXEC_FILE=train_BLIP2IDC.py

echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at $(date +"%T, %d-%m-%Y")
python $EXEC_FILE --model_type "opt" --lora_rank 8 \
    --TRAIN "False" --TEST "True" --save_dir "dir" --little_name "name" \
    --ckpt "weights/spot_sota/spot_1e-4_rank_8_seed_1234__8_15.bin.opt" \
    --lr 4e-5 --module_to_ft "['LLM',QFormer,'ViT']" --vit_embeddings "False" --dataset "spot" \
    --synth_pretraining "False" --batch_size 32
done


