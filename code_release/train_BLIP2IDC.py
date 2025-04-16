import copy
import json
import pickle
from collections import defaultdict
import random
import pandas
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
import torch
from torch import nn, optim
import argparse
import os
from datasets import load_dataset
from collections import defaultdict
import cProfile
from torch.nn import init
from torch.utils.data import Dataset, Sampler
from PIL import Image
from torchvision.transforms import ToTensor, transforms
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, ViTModel, OPTForCausalLM, ViTImageProcessor, ViTFeatureExtractor, \
    AutoImageProcessor, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, VisionEncoderDecoderConfig, \
    VisionEncoderDecoderModel, Blip2Processor, Blip2ForConditionalGeneration, Blip2VisionModel, CLIPProcessor, \
    CLIPModel
import time
from torch.utils.data import DataLoader, Subset
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from utils.eval_utils import *
import logging
import json
import os
import random
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import pandas
import torch
from PIL import Image
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, transforms
from tqdm import tqdm
from transformers import Blip2Processor, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from IDC_datasets import ClevrChangeDataset, SpotTheDiff, ImageEditingRequest, ClevrDC_Dataset, EmuDataset
global logger
from torchvision import transforms
import warnings

MAIN_DIR = "your_dir"

# Filter out the specific warning about max_new_tokens and max_length
warnings.filterwarnings("ignore", message="Both `max_new_tokens` .* and `max_length`.* seem to have been set.*")

def train(model, dataloader, optimizer, device, args):
    model.main_model.to(device)
    model.main_model.train()
    for inputs, labels, idi,idxs in tqdm(dataloader, total=len(dataloader),
                                    bar_format='{l_bar}\033[31m{bar}\033[0m| {n_fmt}/{total_fmt} [{remaining}]'):

        if args.dataset == "IER":
            selected_labels = [min(label_set, key=len) for label_set in labels]
        else:
            selected_labels = [random.choice(label_set) for label_set in labels]
            
        if args.dataset == "emu":
            selected_labels = labels

        input_ids = torch.tensor(
            [model.tokenizer.encode(pr, add_special_tokens=True, max_length=30, padding='max_length',
                                    truncation=True)
             for pr in selected_labels]
        ).cuda()
        
        outputs = model.main_model(input_ids=input_ids.cuda(),
                                   pixel_values=inputs.cuda(),
                                   labels=torch.clone(input_ids).cuda())
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main_clevr(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = initialize_model_with_lora(args=args)
    dataset_semantic_change = []
    dataset_no_change = []
    modules_to_save = []
    best_score_path = f"{MAIN_DIR}/best_Cider_score.txt"

    concat_mode = args.concat_mode
    if args.dataset == "clevr":
        img_dir = f"{MAIN_DIR}/clevr_dataset/data/data/images"
        sc_dir = f"{MAIN_DIR}/clevr_dataset/data/data/sc_images"
        nsc_dir = f"{MAIN_DIR}/clevr_dataset/data/data/nsc_images"

        dataset_semantic_change = ClevrChangeDataset(img_dir=img_dir, modified_img_dir=sc_dir,
                                                     processor=model.processor,
                                                     model_type="opt", data_pair="modified")
        dataset_no_change = ClevrChangeDataset(img_dir=img_dir, modified_img_dir=nsc_dir,
                                               processor=model.processor,
                                               model_type="opt", data_pair="unchanged")
        print("clevr dataset used")
        if dataset_no_change != []:
            concatenated_dataset = ConcatDataset([dataset_semantic_change, dataset_no_change])
        else:
            concatenated_dataset = dataset_semantic_change
        with open(f"{MAIN_DIR}/clevr_dataset/data/data/splits.json", "r") as f:
            split_info = json.load(f)

        train_idx = split_info['train']
        train_nc_idx = [x + len(dataset_semantic_change) for x in train_idx]
        train_data = Subset(concatenated_dataset, train_idx)
        train_nc_data = Subset(concatenated_dataset, train_nc_idx)
        train_dataset = ConcatDataset([train_data, train_nc_data])

        train_all_idx = train_idx + train_nc_idx


        # For validation data
        val_idx = split_info['val']
        val_nc_idx = [x + len(dataset_semantic_change) for x in val_idx]
        val_data = Subset(concatenated_dataset, val_idx)
        val_nc_data = Subset(concatenated_dataset, val_nc_idx)
        val_dataset = ConcatDataset([val_data, val_nc_data])

        test_idx = split_info['test']  # only accounting for the first dataset
        test_nc_idx = [x + len(dataset_semantic_change) for x in test_idx]
        test_data = Subset(concatenated_dataset, test_idx)
        test_nc_data = Subset(concatenated_dataset, test_nc_idx)
        test_dataset = ConcatDataset([test_data, test_nc_data])
        test_all_idx = test_idx + test_nc_idx

        sub_train_loader = DataLoader(subset_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True,
                                      num_workers=32,
                                      pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True,
                                  num_workers=32,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                num_workers=32,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                 num_workers=32,
                                 pin_memory=True)

    if args.dataset == "spot":
        print("DATASET : SPOT")
        img_dir = f"{MAIN_DIR}/spot_the_diff/resized_images/original"
        sc_dir = f"{MAIN_DIR}/spot_the_diff/resized_images/modified"


        train_set = SpotTheDiff(img_dir=img_dir, modified_img_dir=sc_dir, processor=model.processor,
                                label_file=f"{MAIN_DIR}/spot_the_diff/reformat_train.json")
        val_set = SpotTheDiff(img_dir=img_dir, modified_img_dir=sc_dir, processor=model.processor,
                              label_file=f"{MAIN_DIR}/spot_the_diff/reformat_val.json")
        test_set = SpotTheDiff(img_dir=img_dir, modified_img_dir=sc_dir, processor=model.processor,
                               label_file=f"{MAIN_DIR}/spot_the_diff/reformat_test.json")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                  num_workers=32,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                num_workers=32,
                                pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                 num_workers=32,
                                 pin_memory=True, drop_last=True)
       
    if args.dataset == "IER":

        img_dir = f"{MAIN_DIR}/IER_dataset/images"

        train_data_original = ImageEditingRequest(img_dir=img_dir, img_dir_synthetic=img_dir,
                                                  processor=model.processor,
                                                  label_file=f"{MAIN_DIR}/IER_dataset/train.json",
                                                  concat_mode=concat_mode)
        val_data = ImageEditingRequest(img_dir=img_dir, processor=model.processor,
                                        label_file=f"{MAIN_DIR}/IER_dataset/valid.json",
                                                concat_mode=concat_mode)
        
        test_data = ImageEditingRequest(img_dir=img_dir, processor=model.processor,
                                                label_file=f"{MAIN_DIR}/IER_dataset/test.json",
                                                concat_mode=concat_mode)
        
        train_loader = DataLoader(train_data_original, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                  num_workers=32,
                                  pin_memory=True, drop_last=True)
        
        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                  num_workers=32,
                                  pin_memory=True, drop_last=True)

        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                 num_workers=32,
                                 pin_memory=True, drop_last=False)

    
       
    if args.dataset == "DC":
        clevr_dc_dataset = ClevrDC_Dataset(processor=model.processor)

        split_json = f"{MAIN_DIR}/clevr_dc/split_dc.json"
        with open(split_json, 'r') as file:
            split_info = json.load(file)

        train_idx = split_info['train']
        train_data = Subset(clevr_dc_dataset, train_idx)

        test_idx = split_info['test']
        test_data = Subset(clevr_dc_dataset, test_idx)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=custom_collate_clevr_dc, shuffle=True,
                                num_workers=32,
                                pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=custom_collate_clevr_dc, shuffle=False,
                                num_workers=32,
                                pin_memory=True, drop_last=True)

    if args.dataset in ["emu","syned"]:
        def custom_collate_emu(batch, processor=model.processor):

            # Initialize lists to store processed data
            double_images = []
            tasks = []
            instructions = []
            idxs = []
            for sample in batch:
                # Load the PNG image using PIL (Pillow)
                image = sample['image'].convert('RGB')
                edited_image = sample['edited_image'].convert('RGB')
                # Get the size of the original image
                original_size = edited_image.size
                new_image = image.resize(original_size, Image.BICUBIC)

                double_img = Image.new('RGB', (new_image.width, new_image.height + edited_image.height))
                double_img.paste(new_image, (0, 0))
                double_img.paste(edited_image, (0, new_image.height))

                double_img = composed_transforms(double_img)
                # double_img.save("double_image.jpg") #for control purpose
                task_prompt = "Describe the differences between the two images:"
                
                #task_prompt = "Describe the differences between the two images:"
                inputs = processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
                    0).half()
                double_images.append(inputs)

                idxs.append(sample["idx"])
                tasks.append(sample['task'])
                instructions.append(sample['instruction'])

            return torch.stack(double_images), instructions, tasks, idxs

        dataset = load_dataset("facebook/emu_edit_test_set_generations",
                               cache_dir=f"{MAIN_DIR}/.cache/huggingface/datasets")
        data = dataset["validation"]
        og_train_data = []
        og_val_data = []
        
        split_file=f"{MAIN_DIR}/emu_dataset/splits.json"
        with open(split_file,"r") as file :
            split_data = json.load(file)
        
        for x in data: 
            if x["idx"] in split_data["train"]:
                og_train_data.append(x)

            if x["idx"] in split_data["validation"]:
                og_val_data.append(x)
        test_data = dataset["test"]
        
        # train_data = EmuDataset(processor=model.processor,split="train")
        # val_data = EmuDataset(processor=model.processor,split="validation")

        
        if args.synth_pretraining == "True": #means pretraining has been done on customized synthetic data before
            print("use ckpt : ", args.synth_pretraining)

            train_loader = DataLoader(og_train_data,batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                      shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
            val_loader   = DataLoader(og_val_data,batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                      shuffle=False, num_workers=32, pin_memory=True, drop_last=True)
        else: # you train on your synthetic augmentation data first
            
            train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                       num_workers=32, pin_memory=True, drop_last=True)
            val_loader = DataLoader(og_val_data, batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                       num_workers=32, pin_memory=True, drop_last=True)#WE USE THE SAME VAL SPLIT AS EMU

        
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=custom_collate_emu, shuffle=False,
                                 num_workers=32, pin_memory=True, drop_last=True)

        # for testing the performances on each data type

        task_subsets = defaultdict(list)

        # Iterate through the dataset and populate subsets
        for item in dataset['test']:
            task_subsets[item['task']].append(item)

        # Function to create a DataLoader for a given subset
        def create_dataloader_for_task(task_data, collate_fn=custom_collate_emu, batch_size=32):
            return DataLoader(task_data, batch_size=batch_size, collate_fn=collate_fn,
                              shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

        # Create a DataLoader for each task
        task_dataloaders = {task: create_dataloader_for_task(data, custom_collate_emu) for task, data in
                            task_subsets.items()}

    optimizer = optim.AdamW(model.main_model.parameters(), lr=args.lr)

    num_epochs = 100  # max_epochs

    # Initialize variables for early stopping
    best_score = 0.00001
    best_output_model_file = "None"
    early_stop = 0
    epochs = 0
    best_val_loss = 100000000000.0
    logger = get_logger("log.txt")
    best_metrics = []
    try:
        with open(f"{MAIN_DIR}/all_run_scores.json", 'r') as file:
            run_scores = json.load(file)
    except FileNotFoundError:
        run_scores = {}
        with open(f"{MAIN_DIR}/all_run_scores.json", 'w') as file:
            json.dump(run_scores, file, indent=4)
    if f"{args.little_name}_{args.lora_rank}" not in run_scores.keys():
        run_scores[args.little_name] = {}
        run_scores[args.little_name]["best_cider_score"] = 0.0
        run_scores[args.little_name][
            "path"] = f"{MAIN_DIR}/{args.save_dir}/{args.little_name}_{args.lora_rank}"

    best_loss = 10.0
    if args.TRAIN == "True":
        print("Beginning training with validation")
        val_loss = []
        ep=0
        for epoch in tqdm(range(num_epochs), total=num_epochs, desc=f'epoch {epochs}'):
            model.main_model.train()

            train(model, train_loader, optimizer, device, args)
            
            avg_loss = validation(model, dataloader=val_loader, device=device, args=args)
            val_loss.append(avg_loss)
            
       
            if avg_loss < best_loss:
                early_stop = 0
                best_loss = avg_loss
                output_model_file = save_model(epoch, model, save_path=run_scores[args.little_name]["path"]+"_"+str(ep)+"_.bin",
                                               logger=logger)
                
                best_output_model_file = output_model_file
                
                with open(f"{MAIN_DIR}/all_run_scores.json", 'w') as file:
                    json.dump(run_scores, file, indent=4)
                
                logger.info(
                    "The best model is: {}".format(best_output_model_file))
            else : 
                early_stop += 1
               
            ep+=1
            if early_stop == 10:
                break        

            if args.TEST == "True":
                if args.TRAIN == "True":
                    args.ckpt = run_scores[args.little_name]["path"]
                    print(args.ckpt)
                #model = initialize_model_with_lora(args=args)
                if args.dataset in ["emu","IER","spot"] :
                    # print("testing task emu")
                    # CIDEr, task_metrics = eval_emu_mono(model.main_model, task_dataloaders=task_dataloaders, device=device,
                    #                                processor=model.processor,
                    #                                args=args)  # , mean_sentence="there is a person walking in the parking lot")
                    # logger.info(f"test scores are {task_metrics}, best CIDEr is {CIDEr}")
                    print(f"TESTING overall on {args.dataset}")
                    CIDEr, metrics = eval_epoch(model.main_model, dataloader=test_loader, device=device, epoch="eval",
                                                logger=logger,
                                                processor=model.processor,
                                                args=args)  # , mean_sentence="there is a person walking in the parking lot")
                    run_scores[args.little_name]["best_cider_score"] = CIDEr
                    print(metrics)
        
                    with open(f"{MAIN_DIR}/all_run_scores.json", 'w') as file:
                        json.dump(run_scores, file, indent=4)
                    
                    logger.info(f"test scores are {metrics}, best CIDEr is {CIDEr}")
                    print("Tested")
        print(args.TEST == "True")
    if args.TEST == "True":
        print("TESTING")
        CIDEr, metrics = eval_epoch(model.main_model, dataloader=test_loader, device=device, epoch="eval",
                                    logger=logger,
                                    processor=model.processor, args=args,)
                                    #mean_sentence="the scene remains the same")
        logger.info(f"test scores are {metrics}, best CIDEr is {CIDEr}")
        print("Tested")
    print("END TRAINING")
    return best_score, best_metrics

composed_transforms = transforms.Compose([
    transforms.GaussianBlur(3, sigma=(0.1, 0.5))
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class BLIP2IDC(nn.Module):
    def __init__(self, model_type='opt', model_path="Salesforce/blip2-opt-2.7b"):
        super().__init__()
        if model_type == 'opt':
            print(f"MODEL PATH : {model_path}")
            self.main_model = Blip2ForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True,
                revision="51572668da0eb669e01a189dc22abe6088589a24",
            ).half().cuda()
            self.processor = Blip2Processor.from_pretrained(model_path,revision="51572668da0eb669e01a189dc22abe6088589a24")

        self.vision_model = self.main_model.vision_model
        self.LM = self.main_model.language_model 

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,revision="51572668da0eb669e01a189dc22abe6088589a24")
        self.tokenizer.pad_token = self.tokenizer.eos_token

def initialize_model_with_lora(args):
    """
    Initialize a VisionEncoderDecoder model and update it with LoRA layers based on given arguments.

    Parameters:

    Returns:
        model (VisionEncoderDecoder): The updated VisionEncoderDecoder model.
    """
    print("Model initialized")
    model = BLIP2IDC(model_type=args.model_type)

    modules = args.module_to_ft
    target_modules = []

    if 'QFormer' in modules:
        qformer_layers = [f"qformer.encoder.layer.{x}.crossattention.attention.query" for x in range(0, 12)] + ["value",
                                                                                                                "key"]
        target_modules = qformer_layers

    if 'ViT' in modules:
        vit_layers = 'qkv'
        target_modules.append(vit_layers)

    if 'LLM' in modules:
        target_modules.append("q_proj")
        target_modules.append("k_proj")
        target_modules.append("v_proj")
    print(type(args.lora_rank), args.lora_rank)
    print("lora_alpha for scaling : ", 1)
    config = LoraConfig(
        r=int(args.lora_rank), # 8 
        lora_alpha=2 * int(args.lora_rank),  
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules
    )

    model.main_model = get_peft_model(model.main_model, config)
    model.main_model.print_trainable_parameters()

    dir_adapter = args.ckpt
    if dir_adapter is not None:
        bin_checkpoint_name = f"{dir_adapter}/adapter_model.bin"
        safetensors_checkpoint_name = f"{dir_adapter}/adapter_model.safetensors"
        
        import os
        from safetensors.torch import load_file
        
        if os.path.exists(bin_checkpoint_name):
            adapters_weights = torch.load(bin_checkpoint_name)
        elif os.path.exists(safetensors_checkpoint_name):
            adapters_weights = load_file(safetensors_checkpoint_name)
        else:
            raise FileNotFoundError(f"No adapter checkpoint found at {dir_adapter}. Expected either adapter_model.bin or adapter_model.safetensors")
        try:
            cross_attention_layers = f"{dir_adapter}/custom_layers.pth"
            adapters_weights_additionnal = torch.load(cross_attention_layers)
            model.main_model.load_state_dict(adapters_weights_additionnal, strict=False)
            print("loaded")
        except Exception as e:
            print(e)
        print(f"peft weights loaded from {args.ckpt}")
        set_peft_model_state_dict(model.main_model, adapters_weights)
    print("MODEL INITIALIZED")
    return model


def validation(model, dataloader, device, args):
    model.main_model.to(device)
    model.main_model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, idi,idxs in tqdm(dataloader, total=len(dataloader),
                                        bar_format='{l_bar}\033[31m{bar}\033[0m| {n_fmt}/{total_fmt} [{remaining}]'):
            # Randomly select one label for each batch element

            if args.dataset == "IER":
                selected_labels = [min(label_set, key=len) for label_set in labels] #we validate on the shortest label, as it is often the one with only one reference
                
            if args.dataset == "emu":
                selected_labels = labels
            else:
                selected_labels = [random.choice(label_set) for label_set in labels]

            # Tokenize and pad the selected labels
            input_ids = torch.tensor(
                [model.tokenizer.encode(pr, add_special_tokens=True, max_length=30, padding='max_length',
                                        truncation=True)
                 for pr in selected_labels]
            ).cuda()
            # where B is the batch size, C is the number of channels, H is the height, and W is the width
            outputs = model.main_model(input_ids=input_ids.cuda(),
                                       pixel_values=inputs.cuda(),
                                       labels=torch.clone(input_ids).cuda())
            total_loss += outputs["loss"]
        avg_loss = total_loss/len(dataloader)
        print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def custom_collate(batch):
    inputs_0_batch = []
    labels_batch = []
    idi_batch = []

    for item in batch:
        inputs_0, labels, idi = item
        inputs_0_batch.append(inputs_0[0])
        labels_batch.append(labels)
        idi_batch.append(idi)

    idi_batch = torch.stack(idi_batch) if isinstance(idi_batch[0], torch.Tensor) else idi_batch
    return torch.stack(inputs_0_batch), labels_batch, idi_batch,idi_batch


def custom_collate_clevr_dc(batch):
    inputs_0_batch = []
    labels_batch = []
    idi_batch = []

    for item in batch:
        inputs_0, labels, idi = item
        inputs_0_batch.append(inputs_0)
        labels_batch.append(labels)
        idi_batch.append(idi)

    idi_batch = torch.stack(idi_batch) if isinstance(idi_batch[0], torch.Tensor) else idi_batch
    return torch.stack(inputs_0_batch), labels_batch, idi_batch


def save_model(epoch, model, save_path, logger):
    # Only save the model it-self
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_model_file = save_path
    optimizer_state_file = save_path
    model.main_model.save_pretrained(output_model_file)

    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger


def init_weights(m):
    if isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.constant_(m.in_proj_bias, 0)
        nn.init.xavier_uniform_(m.out_proj.weight)
        nn.init.constant_(m.out_proj.bias, 0)




def collate_fn(batch):
    # Find the maximum length string in the batch
    max_len = max(len(s) for s, _ in batch)

    # Initialize list for padded strings and labels
    padded_strings = []
    labels = []

    # Pad strings and copy labels
    for s, label in batch:
        padded_s = s.ljust(max_len, ' ')  # Padding with spaces
        padded_strings.append(padded_s)
        labels.append(label)

    return padded_strings, labels


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    seed = 42  
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"WITH SEED {seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Vision Encoder-Decoder Model Selection")
    parser.add_argument('--model_type', type=str, default='opt',
                        help='Type of model to use. Options are: "opt"')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='rank of the low rank adaptation matrix. Int between 1 to 32')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to the checkpoint to restart from. Default is None')
    parser.add_argument('--TRAIN', type=str, default=True,
                        help='wanna train the model from ckpt ?')
    parser.add_argument('--TEST', type=str, default=True,
                        help='wanna test the model from ckpt ?')
    parser.add_argument('--save_dir', type=str, default="blip2_ckpt",
                        help='where to store the checkpoints')
    parser.add_argument('--little_name', type=str, default="pytorch_model",
                        help="the ckpt are stored as such : save_dir/little_name.bin.epoch")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='select the learning rate for finetuning, between 1e-4 and 1e-5')
    parser.add_argument("--module_to_ft", type=str, default=['ViT', 'QFormer', 'LLM'],
                        help=" which module to finetune using LoRA ? available : ViT, QFormer, LLM")
    parser.add_argument("--vit_embeddings", type=str, default=False,
                        help=" whether to extract the embeddings from the ViT")
    parser.add_argument("--dataset", type=str,
                        help="clevr, spot, IER, Emu")
    parser.add_argument("--synth_pretraining", type=str, default=True,
                        help="do your pretrain is based on synthetic data ?")
    parser.add_argument("--concat_mode", type=str, default="vertical",
                        help="vertical or horizontal concatenation of inputs")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b",revision="51572668da0eb669e01a189dc22abe6088589a24")

    _, _ = main_clevr(args)

