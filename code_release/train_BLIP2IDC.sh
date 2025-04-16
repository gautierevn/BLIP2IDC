#!/usr/bin/env bash


EXEC_FILE=train_BLIP2IDC.py

echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at $(date +"%T, %d-%m-%Y")
python $EXEC_FILE --model_type "opt" --lora_rank 32 \
    --TRAIN "True" --TEST "True" --save_dir "ckpt" --little_name "spot_the_diff" \
    --lr 1e-4 --module_to_ft "['LLM','QFormer','ViT']" --vit_embeddings "False" --dataset "spot" \
    --synth_pretraining "True" --batch_size 48
done


