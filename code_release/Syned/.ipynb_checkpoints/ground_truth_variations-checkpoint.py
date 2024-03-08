import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import tqdm
import json
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset

def standardize(model, sys, user):
    
    if sys:
        return f"[INST] <<SYS>> {sys} <</SYS>> {user} [/INST]"
    else:
        return f"[INST] {user} [/INST]"

def custom_collate_emu(batch):

    instructions = []
    idxs = []
    for sample in batch:
        idxs.append(sample['idx'])
        instructions.append(sample['instruction'])

    return instructions, idxs
    
def get_gt():
        dataset = load_dataset("facebook/emu_edit_test_set_generations",
                               cache_dir="/your_dir/.cache/huggingface/datasets")
        data = dataset["validation"]
        test_data = dataset["test"]
        train_loader = DataLoader(data, batch_size=1, collate_fn=custom_collate_emu, shuffle=False,
                             num_workers=1, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=1, collate_fn=custom_collate_emu, shuffle=False,
                                 num_workers=1, pin_memory=True, drop_last=True)
        train_instructions = []
        # Iterate over test_loader and write instructions with idx into a text file
        with open('instructions_with_idx_train_split.txt', 'w') as file:
            for instructions, idxs in tqdm.tqdm(train_loader, total=len(train_loader)):
                for instruction, idx in zip(instructions, idxs):
                    line = f"{idx} <INSTRUCT> {instruction}"
                    file.write(line + '\n')
        print("saved")

if __name__ == "__main__":
    batch_size = 32 

    torch.set_default_device('cuda')
    device = 'cuda'
    model_path = '/your_dir/Llama-2-7b-chat-hf'
    cache_dir = model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,  low_cpu_mem_usage = True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    sys_prompt = "your goal is to write a variation of the user prompt while keeping the meaning of the instruction. Keep the format of a neutral instruction for a diffusion model. Be concise. Stay really close to the instruction I give you. Do not include any additional information other than the expected output. The expected format is: {variation}"

    results = {}  # Dictionary to store the outputs for each instruction
    instruct_file = 'instructions_with_idx_train_split.txt' #"unique_instructions.txt"

    with open(instruct_file, 'r') as file:
        line_count = sum(1 for line in file)
    #user = "INSTRUCTION : Add the word 'elephants' to the top of the image."
    
    with open(instruct_file, 'r') as file:
        for line in tqdm.tqdm(file,total=line_count):
            # Remove any trailing newlines or spaces
            idx, instruction = line.strip().split("<INSTRUCT>")
            print(idx,instruction)
            instruction = line.strip()
            # Format the string as specified
            user = f"INSTRUCTION : {instruction}"
            prompt = standardize(model,sys_prompt, user)
            inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
            outputs = model.generate(**inputs, max_length=len(inputs[0])+50)
            text = tokenizer.batch_decode(outputs)[0]
            text = text.replace(prompt,'')
            split_text = [line.strip().replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('</s>','') for line in text.split('\n')]
            split_text = [x for x in split_text if "<s>" not in x]
            results[idx] = split_text

            print(split_text)
            
    # Save results to JSON file
    with open('gt_augmented_train.json', 'w') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)