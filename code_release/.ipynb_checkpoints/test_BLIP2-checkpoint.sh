#!/usr/bin/env bash

source /your_venv/bin/activate 

EXEC_FILE=/code_release/train_BLIP2.py

echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at $(date +"%T, %d-%m-%Y")
python $EXEC_FILE --model_type "opt" --lora_rank 32 \
    --TRAIN "False" --TEST "True" --save_dir "dir" --little_name "name" --select_rate 8 \
    --ckpt "the/model/you/want/to/eval.bin.opt"
    --lr 4e-5 --module_to_ft "['LLM',QFormer,'ViT']" --vit_embeddings "False" --dataset "emu" \
    --synth_pretraining "False" --batch_size 32
done


