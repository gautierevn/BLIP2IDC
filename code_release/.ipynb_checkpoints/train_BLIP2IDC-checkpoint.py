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

try:
    from transformers import Blip2ForConditionalGeneration2Images
except ImportError:
    pass
from torch.utils.data import DataLoader, Subset
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
import logging
from cococaption.pycocoevalcap.evil import COCOEvilCap

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.eval import PTBTokenizer, Bleu, Meteor, Rouge, Cider
from fixing_metrics import compute_caption_scores
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

global logger
from torchvision import transforms

composed_transforms = transforms.Compose([
    transforms.GaussianBlur(3, sigma=(0.1, 0.5))
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomHorizontalFlip(p=0.1)
    # Add the JPEG compression here if available in your torchvision version
])
# seed = 1234
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.manual_seed(seed)
# np.random.seed(seed)

import time

class EvalEvil:
    def __init__(self):
        pass

    def eval(self, gt, pred):
        pass

    def eval_batch(self, gt, pred, real=True):
        assert gt.shape[0] == pred.shape[0]
        batch_size = gt.shape[0]
        result = np.zeros([batch_size], np.float32)
        for i in range(batch_size):
            result[i] = self.eval(gt, pred)
        return result


class LanguageEval(EvalEvil):
    def __init__(self):
        self.cocoEvil = COCOEvilCap()

    def eval_whole(self, gt, pred, **kwargs):
        import copy
        self.cocoEvil.evaluate(gt, pred, **kwargs)
        return copy.copy(self.cocoEvil.eval)

    def eval_batch(self, gt, pred, metric=None):
        """
        metric:
        :param gt:
        :param pred:
        :param metric: one of [Bleu_1, ..., Bleu_4, METEOR, ROUGE_L, CIDEr]
        :return:
        """
        self.cocoEvil.evaluate(gt, pred, {metric})
        result = np.zeros(len(gt), np.float32)
        for i in list(self.cocoEvil.imgToEval.keys()):
            result[i] = self.cocoEvil.imgToEval[i][metric]
        return result


def find_mean_sentence_semantic(corpus):
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Calculate the average sentence length
    avg_length = np.mean([len(sentence.split()) for sentence in corpus])

    # Filter sentences that are longer than the average
    filtered_corpus = [sentence for sentence in corpus if len(sentence.split()) <= avg_length]

    # Generate embeddings for each filtered sentence
    sentence_embeddings = model.encode(filtered_corpus)

    # Calculate the centroid of the filtered sentence vectors
    centroid = np.mean(sentence_embeddings, axis=0)

    # Compute term frequencies for the corpus
    word_list = ' '.join(filtered_corpus).split()
    word_freq = Counter(word_list)

    # Compute similarities between centroid and sentence embeddings
    similarities = util.pytorch_cos_sim(centroid, sentence_embeddings)[0]

    # Compute TF-weighted similarities
    tf_weighted_similarities = []
    for idx, sentence in enumerate(filtered_corpus):
        tf_sum = sum(word_freq[word] for word in sentence.split())
        tf_weighted_similarity = similarities[idx].item() + tf_sum
        tf_weighted_similarities.append(tf_weighted_similarity)

    # Find the sentence whose vector is closest to the centroid and has the most frequent words
    closest_sentence_index = np.argmax(tf_weighted_similarities)
    closest_sentence = filtered_corpus[closest_sentence_index]
    print(closest_sentence)
    return closest_sentence


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper


class EvalCap(COCOEvalCap):
    def __init__(self, coco, cocoRes):
        super(EvalCap, self).__init__(coco, cocoRes)

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]
        print("gts", len(gts), "res", len(res))
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()


def score_generation(anno_file, result_file):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)
    coco_eval = EvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    # print(coco_eval.params['image_id'])
    coco_eval.evaluate()
    return copy.deepcopy(coco_eval.eval)


def score_generation_by_type(anno_file, result_file, type_file):
    print("GENERATION by type")
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)

    coco_eval = EvalCap(coco, coco_res)

    type_dict = json.load(open(type_file, 'r'))
    results = {}
    print(len(coco_res.getImgIds()))
    for type, image_ids in type_dict.items():
        filtered = set(coco_res.getImgIds()).intersection(set(image_ids))
        coco_eval.params['image_id'] = list(filtered)
        coco_eval.evaluate()
        results[type] = copy.deepcopy(coco_eval.eval)

    return results


def add_distractor_type_(type_dict_path):
    with open(type_dict_path, 'r') as f:
        type_dict = json.load(f)
    # Aggregate all unique image_ids from existing types
    all_image_ids = set()
    for image_ids in type_dict.values():
        all_image_ids.update(image_ids)
    print(len(all_image_ids))
    # Create new image_ids for "distractor" by appending "_n" to the aggregated unique image_ids
    distractor_image_ids = [f"{image_id}_n" for image_id in all_image_ids]
    print(len(distractor_image_ids))
    # Add the new "distractor" type to the dictionary
    type_dict["distractor"] = distractor_image_ids
    print(len(type_dict.values()))
    with open("your_DIR/clevr_dataset/data/data/type_mapping_with_distractors.json", 'w') as f:
        json.dump(type_dict, f, indent=4)


def eval_epoch(model, dataloader, device, epoch=None, logger=None, processor=None, args=None, mean_sentence=None):
    model = model.to(device)
    model.eval()

    all_result_lists = []
    gt_list = []
    change_results_list = []
    no_change_results_list = []
    embeds = []
    with open("your_DIR/emu_dataset/gt_augmented_test_clean.json", 'r') as json_file:
        emu_gt_test = json.load(json_file)
        
    original_gt={}
    with open("emu_dataset/instructions_with_idx_test_split.txt", 'r') as file:
        for line in file:
            # Remove any trailing newlines or spaces
            idx, instruction = line.strip().split("<INSTRUCT>")
            original_gt[idx]=instruction
    #print(original_gt.keys())
    if mean_sentence != None:  # for baseline purpose
        print("Mean sentence used :", mean_sentence)
    if args.architecture == "mono":
        with torch.no_grad():
            print('using mono architecture for testing')
            for inputs_0, labels, idi,idxs in tqdm(dataloader, total=len(dataloader)):
                input = inputs_0[0]
                if len(inputs_0) == 2:
                    continue
                else:

                    generated_ids = model.generate(inputs_0.cuda(), max_new_tokens=77)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                # Generate a unique image_id (for example, incrementing an existing counter)
                image_id = idi
                # Prepare data for metrics calculation
                labels_possible_list = ["the scene remains the same", "nothing has changed", "nothing was modified",
                                        "the two scenes seem identical", "the scene is the same as before",
                                        "no change was made", "no change has occured", "there is no change"]
                for c, id in enumerate(idxs):
                    if bool(set(labels_possible_list) & set(labels[c])) and generated_text[c] in labels_possible_list:
                        generated_text[c] = random.choice(labels[c])
                        no_change_results_list.append({"caption": generated_text[c], "image_id": id})
                    elif bool(set(labels_possible_list) & set(labels[c])):
                        no_change_results_list.append({"caption": generated_text[c], "image_id": id})
                    else:
                        change_results_list.append({"caption": generated_text[c], "image_id": id})
                    if mean_sentence != None:
                        all_result_lists.append({"caption": mean_sentence, "image_id": id})
                    else:
                        if args.dataset == "IER" or args.augmentation == "IER" or args.dataset == "magic" or args.dataset == "emu" or args.dataset == "DC":
                            all_result_lists.append(generated_text[c])
                            if args.dataset !="emu":                             
                                gt_list.append(labels[c])
                            else : 
                                gt_list.append(emu_gt_test[f'{id} ']+[original_gt[f'{id} ']])#clean dataset
                        else:
                            all_result_lists.append({"caption": generated_text[c], "image_id": id})

    if args.dataset == "clevr":
        anno_file = "your_DIR/clevr_dataset/data/data/total_change_captions_reformat.json"

    if args.dataset == "magic":
        print("use the other script in VisualRelationships for magic dataset")
        total_results = all_result_lists
        json.dump(total_results, open(f"results/results_MB_on_MB.json", "w"))
        preds = json.load(open('results/results_MB_on_MB.json'))
        gts = []
        preds_list = []
        sents = []
        for inputs, labels, idi in tqdm(dataloader, total=len(dataloader),
                                        bar_format='{l_bar}\033[31m{bar}\033[0m| {n_fmt}/{total_fmt} [{remaining}]'):
            sents.append(labels)  # 32 batch

        for label in sents:
            for i in label:
                gts.append(i)

        for datum in preds:
            preds_list.append(datum)

        langeval = LanguageEval()
        metrics = langeval.eval_whole(gts, preds_list)
        return metrics["CIDEr"], metrics

    if args.dataset == "IER" or args.augmentation == "IER" or args.dataset == "emu" or args.dataset == "DC":
        print(f"evaluating {args.dataset}")
        print("use the other script in VisualRelationships")
        total_results = all_result_lists
        json.dump(total_results, open(f"your_DIR/results/results_MB_on_{args.dataset}.json", "w"))
        dataset = json.load(open("your_DIR/IER_dataset/test.json"))
        preds = json.load(open(f'your_DIR/results/results_MB_on_{args.dataset}.json'))
        gts = []
        preds_list = []
        for datum in dataset:
            sents = datum['sents']
            gts.append(sents)
        for datum in preds:
            preds_list.append(datum)
        print(preds_list[0:5], gts[0:5])
        langeval = LanguageEval()
        if args.dataset == "emu" or args.dataset == "DC":
            metrics = langeval.eval_whole(gt_list, preds_list)
            return metrics["CIDEr"], metrics
        metrics = langeval.eval_whole(gts, preds_list)
        print(metrics)
        return metrics["CIDEr"], metrics
    if args.dataset == "spot":
        anno_file = "your_DIR/spot_the_diff/valid_part.json"
    # Combine results for metrics calculation
    total_results = all_result_lists
    # print("TOTAL RESULTS",total_results)
    json.dump(change_results_list, open(f"results/sc_results.json", "w"))
    json.dump(no_change_results_list, open(f"results/nsc_results.json", "w"))
    json.dump(total_results, open(f"results/hyp_ep_{epoch}_FULL.json", "w"))

    assert os.path.exists(f"results/hyp_ep_{epoch}_FULL.json")

    # Evaluate
    result_file = f"results/hyp_ep_{epoch}_FULL.json"

    metrics_nlg = score_generation(anno_file=anno_file, result_file=result_file)
    # Logging the metrics
    logger.info(
        f">>>  BLEU_1: {metrics_nlg['Bleu_1']:.4f}, BLEU_2: {metrics_nlg['Bleu_2']:.4f}, BLEU_3: {metrics_nlg['Bleu_3']:.4f}, BLEU_4: {metrics_nlg['Bleu_4']:.4f}")
    logger.info(
        f">>>  METEOR: {metrics_nlg['METEOR']:.4f}, ROUGE_L: {metrics_nlg['ROUGE_L']:.4f}, CIDEr: {metrics_nlg['CIDEr']:.4f}")

    CIDEr = metrics_nlg["CIDEr"]
    return CIDEr, metrics_nlg


def eval_emu_mono(model, task_dataloaders, device, processor, args, mean_sentence=None):
    model = model.to(device)
    model.eval()
    with open("your_DIR/emu_dataset/gt_augmented_test_clean.json", 'r') as json_file:
        emu_gt_test = json.load(json_file)
    original_gt={}
    
    with open("emu_dataset/instructions_with_idx_test_split.txt", 'r') as file:
        for line in file:
            # Remove any trailing newlines or spaces
            idx, instruction = line.strip().split("<INSTRUCT>")
            original_gt[idx]=instruction
    
    # Initialize results storage for each task
    all_results_by_task = {}
    metrics_by_task = {}
    with torch.no_grad():
        print('Evaluating EMU dataset using mono architecture')

        for task, dataloader in task_dataloaders.items():
            print(f"Evaluating task: {task}")
            all_result_lists = []
            gt_list = []
            change_results_list = []
            no_change_results_list = []
            embeds = []

            for inputs_0, labels, idi, id_number in tqdm(dataloader, total=len(dataloader)):
                input = inputs_0[0]
                if len(inputs_0) == 2:
                    continue
                else:
                    if False:
                        image_embeds = model.vision_model(inputs_0.cuda(), return_dict=True).last_hidden_state
                        embeds.append(image_embeds)
                    generated_ids = model.generate(inputs_0.cuda(), max_new_tokens=77)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

                # Labels processing for EMU dataset
                labels_possible_list = ["the scene remains the same", "nothing has changed", "nothing was modified",
                                        "the two scenes seem identical", "the scene is the same as before",
                                        "no change was made", "no change has occured", "there is no change"]
                for c, id in enumerate(id_number):#idi before
                    if bool(set(labels_possible_list) & set(labels[c])) and generated_text[c] in labels_possible_list:
                        generated_text[c] = random.choice(labels[c])
                        no_change_results_list.append({"caption": generated_text[c], "image_id": id})
                    elif bool(set(labels_possible_list) & set(labels[c])):
                        no_change_results_list.append({"caption": generated_text[c], "image_id": id})
                    else:
                        change_results_list.append({"caption": generated_text[c], "image_id": id})
                    if mean_sentence != None:
                        all_result_lists.append({"caption": mean_sentence, "image_id": id})
                    else:
                        all_result_lists.append(generated_text[c])

                        gt_list.append(emu_gt_test[f'{id} ']+[original_gt[f'{id} ']])

            # Store results for the current task
            all_results_by_task[task] = {
                "all_result_lists": all_result_lists,
                "gt_list": gt_list,
            }

        if args.dataset == "IER" or args.augmentation == "IER" or args.dataset == "emu":
            print(f"evaluating {args.dataset}")
            print("use the other script in VisualRelationships")
            for task, all_result_lists in all_results_by_task.items():
                total_results = all_result_lists["all_result_lists"]
                gt_list = all_result_lists["gt_list"]

                json.dump(total_results, open(f"results/results_MB_on_{args.dataset}.json", "w"))
                dataset = json.load(open("your_DIR/IER_dataset/test.json"))
                preds = json.load(open(f'your_DIR/results/results_MB_on_{args.dataset}.json'))
                gts = []
                preds_list = []
                for datum in dataset:
                    sents = datum['sents']
                    gts.append(sents)
                for datum in preds:
                    preds_list.append(datum)

                langeval = LanguageEval()
                if args.dataset == "emu":
                    metrics = langeval.eval_whole(gt_list, preds_list)
                    metrics_by_task[task] = metrics

                    print(task, metrics_by_task[task])
        # return metrics_by_task
    json.dump(metrics_by_task, open(f"augmentation_result/emu_task_results.json", "w"))

    return metrics_by_task, 1


# Function to store relevant elements for later use
def store_generated_and_reference_texts(processor, generated_ids, labels, storage_dict):
    """
    Store generated and reference texts along with image IDs into a dictionary for later use.

    Parameters:
    - processor: Text processing object (for decoding generated IDs)
    - generated_ids: IDs generated by some model (to be decoded into text)
    - labels: Labels containing the reference texts
    - storage_dict: Dictionary where the generated and reference texts are stored

    Returns:
    - Updated storage_dict
    """
    # Decode the generated IDs to text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Randomly select one reference text from labels
    ref_text = random.choice(labels)[0]

    # Generate a unique image_id (for example, incrementing an existing counter)
    image_id = len(storage_dict.get('annotations', [])) + 1

    # Update the storage_dict with the new data
    storage_dict.setdefault('annotations', []).append({'image_id': image_id, 'caption': generated_text})
    storage_dict.setdefault('references', []).append({'image_id': image_id, 'caption': ref_text})

    with open('pycoco_dict_results.json', 'w') as f:
        json.dump(storage_dict, f)
    return storage_dict


class VisionEncoderDecoder(nn.Module):
    def __init__(self, model_type='opt', model_path="Salesforce/blip2-opt-2.7b"):
        super().__init__()
        print("loading main model")

        if model_type == 'opt':
            print(f"MODEL PATH : {model_path}")
            self.main_model = Blip2ForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True
            ).half().cuda()
            self.processor = Blip2Processor.from_pretrained(model_path)

        elif model_type == 'flan':
            model_path = "Salesforce/blip2-flan-t5-xl"
            print(f"MODEL PATH : {model_path}")
            self.main_model = Blip2ForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True
            ).cuda().half()
            self.processor = Blip2Processor.from_pretrained(model_path)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        self.vision_model = self.main_model.vision_model
        self.LM = self.main_model.language_model  # decoder to text

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def decode_images(self, hidden_states, att_mask, labels):
        # Input: Hidden states
        # Output: Modified image
        text_model = self.LM
        outputs = text_model(input_ids=labels, labels=labels,
                             encoder_hidden_states=hidden_states)  # , return_dict=True, labels=labels)
        return outputs

class ClevrChangeDataset(Dataset):
    def __init__(self, img_dir, modified_img_dir, processor, transform=None,
                 label_file='your_DIR/clevr_dataset/data/data/total_change_captions_reformat.json',
                 model_type=None, data_pair="modified"):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.mod_img_dir = modified_img_dir
        self.transform = transform
        self.processor = processor
        self.model_type = model_type
        self.data_pair = data_pair
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.annotations_by_image = {}
        self.captions_by_image = {}
        for annotation in self.labels["annotations"]:
            image_id = annotation["image_id"]
            caption = annotation["caption"]

            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
                self.captions_by_image[image_id] = []

            self.annotations_by_image[image_id].append(annotation)
            self.captions_by_image[image_id].append(caption)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        if self.data_pair == "modified":
            mod_img_path = img_path.replace('images', 'sc_images').replace('default', 'semantic')
        else:
            mod_img_path = img_path.replace('images', 'nsc_images').replace('default', 'nonsemantic')
        label_id = os.path.basename(img_path)
        idi = label_id.split("_")[2]

        if self.data_pair != "modified":
            idi = idi + "_n"
        try:
            labels = self.captions_by_image.get(idi, [])
        except KeyError or FileNotFoundError:
            print("ERROR")
        img_0 = Image.open(img_path)
        img_1 = Image.open(mod_img_path)

        double_img = Image.new('RGB', (img_0.width, img_0.height + img_1.height))
        double_img.paste(img_0, (0, 0))
        double_img.paste(img_1, (0, img_0.height))
        task_prompt = "Describes the differences between the image on the top and the image at the bottom :"
        inputs = self.processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
            0).half()
        if self.model_type == "instructblip":
            qformer_inputs_ids = self.processor(images=double_img, text=task_prompt, return_tensors="pt")[
                "qformer_input_ids"].squeeze(0)
            return [inputs, qformer_inputs_ids], labels, idi
        return [inputs], labels, idi


class SpotTheDiff(Dataset):
    def __init__(self, img_dir, modified_img_dir, processor, transform=None,
                 label_file='your_DIR/spot_the_diff/reformat_train.json',
                 model_type=None):
        self.img_dir = img_dir
        self.mod_img_dir = modified_img_dir
        self.transform = transform
        self.processor = processor
        self.model_type = model_type
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.img_names = []
        self.sentences = []
        for i in self.labels:
            self.img_names.append(i["img_id"])
            self.sentences.append(i["sentences"])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_id = self.img_names[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        mod_img_path = os.path.join(self.mod_img_dir, f"{img_id}_2.png")

        idi = self.img_names[idx] + ".png"
        labels = self.sentences[idx]

        img_0 = Image.open(img_path)
        img_1 = Image.open(mod_img_path)
        task_prompt = "Describes the differences between the image on the top(left) and the image at the bottom(right) :"

        double_img = Image.new('RGB', (img_0.width, img_0.height + img_1.height))
        double_img.paste(img_0, (0, 0))
        double_img.paste(img_1, (0, img_0.height))
        inputs = self.processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
            0).half()
        return [inputs], labels, idi


class ImageEditingRequest(Dataset):
    def __init__(self, img_dir, img_dir_synthetic, processor, transform=None,
                 label_file='your_DIR/IER_dataset/train.json',
                 model_type=None, concat_mode='vertical'):  # Added concat_mode
        self.img_dir = img_dir  # "your_DIR/IER_dataset/images"
        self.img_dir_synthetic = img_dir_synthetic  # 'your_DIR/instruct-pix2pix/generated_images_512'

        self.transform = transform
        self.processor = processor
        self.model_type = model_type
        self.concat_mode = concat_mode  # New parameter
        self.label_file = label_file
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.img_names_0 = []
        self.img_names_1 = []
        self.sentences = []

        self.uids = []

        for i in self.labels:
            self.img_names_0.append(i["img0"])
            self.img_names_1.append(i["img1"])
            self.sentences.append(i["sents"])
            self.uids.append(i["uid"])

    def __len__(self):
        return len(self.img_names_0)

    def __getitem__(self, idx):
        img_id_0 = self.img_names_0[idx]
        img_id_1 = self.img_names_1[idx]
        img_path_0 = os.path.join(self.img_dir, img_id_0)
        if self.label_file == 'your_DIR/IER_dataset/test.json':
            img_path_1 = os.path.join(self.img_dir, img_id_1)
        else:
            img_path_1 = os.path.join(self.img_dir_synthetic, img_id_1)

        labels = self.sentences[idx]
        uid = self.uids[idx]
        img_0 = Image.open(img_path_0)
        img_1 = Image.open(img_path_1)
        original_size = img_1.size
        img_0 = img_0.resize(original_size, Image.BICUBIC)

        if self.concat_mode == 'vertical':
            double_img = Image.new('RGB', (img_0.width, img_0.height + img_1.height))
            double_img.paste(img_0, (0, 0))
            double_img.paste(img_1, (0, img_0.height))
        elif self.concat_mode == 'horizontal':
            double_img = Image.new('RGB', (img_0.width + img_1.width, img_0.height))
            double_img.paste(img_0, (0, 0))
            double_img.paste(img_1, (img_0.width, 0))
        else:
            raise ValueError("Invalid concat_mode. Choose either 'vertical' or 'horizontal'.")

        double_img = composed_transforms(double_img)

        task_prompt = "Describe the differences between the two images:"
        inputs = self.processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
            0).half()
        return [inputs], labels, uid, uid


class ClevrDC_Dataset(Dataset):
    def __init__(self, processor, json_file="your_DIR/BIG_storage/clevr_dc/captions_dc.json",
                 concat_mode='vertical', data_path="your_DIR/BIG_storage/clevr_dc"):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.to_tensor = ToTensor()
        self.processor = processor
        self.concat_mode = concat_mode  # New parameter
        self.data_path = data_path
        self.id = list(self.data.keys())
        # print(self.id[0])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index = self.id[idx]
        captions = self.data[index]

        source_img_path = os.path.join(self.data_path, f"bef/{index}.png")
        target_img_path = os.path.join(self.data_path, f"aft/{index}.png")

        source_img = Image.open(source_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')

        double_img = Image.new('RGB', (source_img.width, source_img.height + target_img.height))
        double_img.paste(source_img, (0, 0))

        double_img.paste(target_img, (0, source_img.height))
        task_prompt = ""
        inputs = self.processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
            0).half()

        return inputs, captions, idx


class MagicDataset(Dataset):
    def __init__(self, data, processor, concat_mode='vertical'):
        self.data = data
        self.to_tensor = ToTensor()
        self.processor = processor
        self.concat_mode = concat_mode  # New parameter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        source_img = sample['source_img']
        img_id = sample['img_id']
        instruction = sample['instruction']
        target_img = sample['target_img']

        if random.random() < 0.05:
            target_img = source_img
            instruction = random.choice(
                ["there is no change", "images are the same", "no manipulation was detected", "no edit is made"])


        if self.concat_mode == 'vertical':
            double_img = Image.new('RGB', (source_img.width, source_img.height + target_img.height))
            double_img.paste(source_img, (0, 0))
            double_img.paste(target_img, (0, source_img.height))
        elif self.concat_mode == 'horizontal':
            double_img = Image.new('RGB', (source_img.width + target_img.width, source_img.height))
            double_img.paste(source_img, (0, 0))
            double_img.paste(target_img, (source_img.width, 0))
        else:
            raise ValueError("Invalid concat_mode. Choose either 'vertical' or 'horizontal'.")

        task_prompt = "Describes the differences between the two images:"
        inputs = self.processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
            0).half()

        return [inputs], instruction, img_id

class EmuDataset(Dataset):
    def __init__(self, processor, split="train",json_file="your_DIR/emu_dataset/augmented_dataset.json", split_file="your_DIR/emu_dataset/splits.json", 
                 concat_mode='vertical', use_distractors=False,
                 ranking_json="your_DIR/emu_filtered/all_clip_scores.json", rank=0):
        with open(json_file, 'r') as file:
            self.dataset = json.load(file)
        with open(ranking_json, 'r') as file:
            self.ranking = json.load(
                file)  # ranking[idx][x]["path"] give you the path of the x+1 best generated images, included emu ones, wrt clip simi
        with open(split_file,"r") as file:
            split_data = json.load(file)
        self.data = []
        for x in self.dataset : 
            if x["idx"] in split_data[split]:
                self.data.append(x)
        self.to_tensor = ToTensor()
        self.processor = processor
        self.concat_mode = concat_mode  # New parameter
        self.use_distractors = use_distractors
        self.rank = rank
        if self.use_distractors:
            print("USING DISTRACTORS")

    def __len__(self):
        return len(self.data)
        # return len(self.ranking)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        source_img_path = sample['image']
        target_img_path = sample['edited_image']

        img_id = sample['idx']
        # rank_sample = self.ranking[str(img_id)]
        # target_img_path = rank_sample[self.rank]['path']
        # instruction = sample['instruction']
        source_img = Image.open(source_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')
        
        sample["image"] = source_img
        sample["edited_image"] = target_img
        return sample


def check_for_nans_or_infs(model):
    print("BEFORE ANYTHING : CHECKING INITIALIZATION")
    modules = ['tcas.1.multihead_attention.in_proj_bias', 'tcas.1.multihead_attention.out_proj.bias',
               'tcas.0.multihead_attention.in_proj_bias', 'tcas.1.multihead_attention.in_proj_weight',
               'tcas.0.multihead_attention.out_proj.bias', 'tcas.0.multihead_attention.in_proj_weight',
               'tcas.0.multihead_attention.out_proj.weight', 'linear_projection.weight',
               'tcas.1.multihead_attention.out_proj.weight', 'linear_projection.bias']

    for name, param in model.named_parameters():
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            print(f"Parameter {name} has NaN or Inf values!")
            # initial_bias = model.main_model.tcas[0].multihead_attn.in_proj_bias.data
            # initial_bias3 = model.main_model.tcas[0].self_attn.in_proj_bias.data

            # Print or log the initial bias values
            # print("Initial bias values:", initial_bias, initial_bias3)

            print(f"Parameter {name} has NaN or Inf values!")

            # Reinitialize the offending parameters
            if name in modules:
                # Zero initialization for multihead attention's in_proj_bias
                with torch.no_grad():
                    param.fill_(1.0)
                print(f"Reinitialized {name} to zeros.")

                print(f"Reinitialized {name} to zeros.")
def validation(model, dataloader, device, args):
    model.main_model.to(device)
    model.main_model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, idi,idxs in tqdm(dataloader, total=len(dataloader),
                                        bar_format='{l_bar}\033[31m{bar}\033[0m| {n_fmt}/{total_fmt} [{remaining}]'):
            # Randomly select one label for each batch element

            if args.dataset == "IER":
                selected_labels = [min(label_set, key=len) for label_set in labels]
            else:
                selected_labels = [random.choice(label_set) for label_set in labels]
            if args.dataset == "magic" or args.dataset == "emu":
                selected_labels = labels
            # Tokenize and pad the selected labels
            input_ids = torch.tensor(
                [model.tokenizer.encode(pr, add_special_tokens=True, max_length=60, padding='max_length',
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


@time_decorator
def train(model, dataloader, optimizer, device, args):
    model.main_model.to(device)
    model.main_model.train()
    # check_for_nans_or_infs(model)
    if args.architecture == "mono":
        for inputs, labels, idi,idxs in tqdm(dataloader, total=len(dataloader),
                                        bar_format='{l_bar}\033[31m{bar}\033[0m| {n_fmt}/{total_fmt} [{remaining}]'):
            # Randomly select one label for each batch element

            if args.dataset == "IER":
                selected_labels = [min(label_set, key=len) for label_set in labels]
            else:
                selected_labels = [random.choice(label_set) for label_set in labels]
            if args.dataset == "magic" or args.dataset == "emu":
                selected_labels = labels
            # Tokenize and pad the selected labels
            input_ids = torch.tensor(
                [model.tokenizer.encode(pr, add_special_tokens=True, max_length=60, padding='max_length',
                                        truncation=True)
                 for pr in selected_labels]
            ).cuda()
            # where B is the batch size, C is the number of channels, H is the height, and W is the width
            outputs = model.main_model(input_ids=input_ids.cuda(),
                                       pixel_values=inputs.cuda(),
                                       labels=torch.clone(input_ids).cuda())
            loss = outputs["loss"]
            # print("Loss : ",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    else:
        for inputs_0, inputs_1, labels, idi in tqdm(dataloader, total=len(dataloader),
                                                    bar_format='{l_bar}\033[31m{bar}\033[0m| {n_fmt}/{total_fmt} [{remaining}]'):
            # Randomly select one label for each batch element

            selected_labels = [random.choice(label_set) for label_set in labels]
            # Tokenize and pad the selected labels
            input_ids = torch.tensor(
                [model.tokenizer.encode(pr, add_special_tokens=True, max_length=30, padding='max_length',
                                        truncation=True)
                 for pr in selected_labels]
            ).cuda()
            outputs = model.main_model(input_ids=input_ids.cuda(),
                                       pixel_values=[inputs_0.cuda(), inputs_1.cuda()],
                                       labels=torch.clone(input_ids).cuda())

            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def validate(model, dataloader, optimizer, device, args):
    model.main_model.to(device)
    model.main_model.eval()

    print("validating")
    # check_for_nans_or_infs(model)

    if args.architecture == "mono":
        with torch.no_grad():
            for inputs, labels, idi, idxs in tqdm(dataloader, total=len(dataloader),
                                            bar_format='{l_bar}\033[31m{bar}\033[0m| {n_fmt}/{total_fmt} [{remaining}]'):
                # Randomly select one label for each batch element
                if args.dataset == "magic":
                    selected_labels = labels
                else:
                    selected_labels = [random.choice(label_set) for label_set in labels]
                # Tokenize and pad the selected labels
                input_ids = torch.tensor(
                    [model.tokenizer.encode(pr, add_special_tokens=True, max_length=60, padding='max_length',
                                            truncation=True)
                     for pr in selected_labels]
                ).cuda()
                # where B is the batch size, C is the number of channels, H is the height, and W is the width
                outputs = model.main_model(input_ids=input_ids.cuda(),
                                           pixel_values=inputs.cuda(),
                                           labels=torch.clone(input_ids).cuda())
                loss = outputs["loss"]


    else:
        for inputs_0, inputs_1, labels, idi in tqdm(dataloader, total=len(dataloader),
                                                    bar_format='{l_bar}\033[31m{bar}\033[0m| {n_fmt}/{total_fmt} [{remaining}]'):
            # Randomly select one label for each batch element

            selected_labels = [random.choice(label_set) for label_set in labels]
            # Tokenize and pad the selected labels
            input_ids = torch.tensor(
                [model.tokenizer.encode(pr, add_special_tokens=True, max_length=30, padding='max_length',
                                        truncation=True)
                 for pr in selected_labels]
            ).cuda()
            outputs = model.main_model(input_ids=input_ids.cuda(),
                                       pixel_values=[inputs_0.cuda(), inputs_1.cuda()],
                                       labels=torch.clone(input_ids).cuda())

            loss = outputs["loss"]
    return loss


def custom_collate(batch):
    inputs_0_batch = []
    labels_batch = []
    idi_batch = []

    for item in batch:
        inputs_0, labels, idi,idi = item
        inputs_0_batch.append(inputs_0[0])
        labels_batch.append(labels)
        idi_batch.append(idi)

    # Assuming inputs_0 and inputs_1 are tensors, you can stack them
    # inputs_0_batch = torch.stack(inputs_0_batch)
    # inputs_1_batch = torch.stack(inputs_1_batch) 

    # Since labels can be of different lengths, we keep it as a list
    # If you want to pad them to have the same length, you can do so here

    # Assuming idi is a scalar or a tensor of the same shape across the batch, you can stack or keep as a list
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

    # Assuming inputs_0 and inputs_1 are tensors, you can stack them
    # inputs_0_batch = torch.stack(inputs_0_batch)
    # inputs_1_batch = torch.stack(inputs_1_batch)

    # Since labels can be of different lengths, we keep it as a list
    # If you want to pad them to have the same length, you can do so here

    # Assuming idi is a scalar or a tensor of the same shape across the batch, you can stack or keep as a list
    idi_batch = torch.stack(idi_batch) if isinstance(idi_batch[0], torch.Tensor) else idi_batch
    return torch.stack(inputs_0_batch), labels_batch, idi_batch


def custom_collate_instructblip(batch):
    inputs_0_batch = []
    qformer_ids_batch = []
    labels_batch = []
    idi_batch = []

    for item in batch:
        inputs, labels, idi = item
        inputs_0_batch.append(inputs[0])
        qformer_ids_batch.append(inputs[1])
        labels_batch.append(labels)
        idi_batch.append(idi)

    # Assuming inputs_0 and inputs_1 are tensors, you can stack them
    # inputs_0_batch = torch.stack(inputs_0_batch)
    # inputs_1_batch = torch.stack(inputs_1_batch)

    # Since labels can be of different lengths, we keep it as a list
    # If you want to pad them to have the same length, you can do so here

    # Assuming idi is a scalar or a tensor of the same shape across the batch, you can stack or keep as a list
    idi_batch = torch.stack(idi_batch) if isinstance(idi_batch[0], torch.Tensor) else idi_batch

    return [torch.stack(inputs_0_batch), torch.stack(qformer_ids_batch)], labels_batch, idi_batch



def computing_properties(hyp_json=f"pycoco_dict_hyp_3e-5_0.json", ref_json=f"pycoco_dict_refs_3e-5_0.json",
                         mapping_json_path="/data/clevr-change/data/type_mapping.json"):
    metrics_dictionnary = {}

    with open(hyp_json, 'r') as f:
        # Load the JSON data
        hypothesis_captions = json.load(f)

    with open(ref_json, 'r') as f:
        # Load the JSON data
        reference_captions = json.load(f)

    # Open the file for reading
    with open(mapping_json_path, 'r') as f:
        # Load the JSON data
        type_mapping = json.load(f)

    hypothesis_annotations = hypothesis_captions['annotations']
    reference_annotations = reference_captions['annotations']

    # Flatten the image_id lists in new_hypothesis_annotations and new_reference_annotations
    flattened_new_hypothesis_annotations = [{'image_id': hyp['image_id'][0], 'caption': hyp['caption']} for hyp in
                                            hypothesis_annotations]
    flattened_new_reference_annotations = [{'image_id': ref['image_id'][0], 'caption': ref['caption']} for ref in
                                           reference_annotations]

    # Collect all unique image IDs from type_mapping
    all_type_mapping_image_ids = set([img_id for img_ids in type_mapping.values() for img_id in img_ids])

    # Now, let's try again to find overlaps and segregate the data accordingly.

    # Collect all unique image IDs from the new, flattened files
    new_all_hypothesis_image_ids = set([hyp['image_id'] for hyp in flattened_new_hypothesis_annotations])
    new_all_reference_image_ids = set([ref['image_id'] for ref in flattened_new_reference_annotations])

    # Check for overlaps
    new_overlap_hypothesis_image_ids = new_all_hypothesis_image_ids.intersection(all_type_mapping_image_ids)
    new_overlap_reference_image_ids = new_all_reference_image_ids.intersection(all_type_mapping_image_ids)

    # Proceed with segregation if there are overlaps
    if new_overlap_hypothesis_image_ids or new_overlap_reference_image_ids:
        # Initialize dictionaries to store the segregated data for each property
        new_segregated_hypothesis = defaultdict(list)
        new_segregated_references = defaultdict(list)

        # Loop through each property type and the associated image_ids
        for property_type, image_ids in type_mapping.items():
            # Loop through each hypothesis annotation
            for hyp in flattened_new_hypothesis_annotations:
                # If the image_id in hypothesis annotations matches one in the current property type
                if hyp['image_id'] in image_ids:
                    # Add to the segregated hypothesis data for the current property type
                    new_segregated_hypothesis[property_type].append(hyp)

            # Loop through each reference annotation
            for ref in flattened_new_reference_annotations:
                # If the image_id in reference annotations matches one in the current property type
                if ref['image_id'] in image_ids:
                    # Add to the segregated references data for the current property type
                    new_segregated_references[property_type].append(ref)

    # Save the segregated JSON files
    new_created_files = []
    for property_type in new_segregated_hypothesis.keys():
        hyp_filename = f'{property_type}_hyp.json'
        ref_filename = f'{property_type}_ref.json'
        with open(hyp_filename, 'w') as f:
            json.dump({"annotations": new_segregated_hypothesis[property_type]}, f)
        with open(ref_filename, 'w') as f:
            json.dump({"annotations": new_segregated_references[property_type]}, f)
        new_created_files.extend([hyp_filename, ref_filename])

    for prop in type_mapping.keys():
        ref_file_path = f"{prop}_ref.json"
        hyp_file_path = f"{prop}_hyp.json"

        scores = compute_caption_scores(ref_file_path=ref_file_path, hypo_file_path=hyp_file_path,
                                        include_metrics=["bleu", "rouge", "cider"])
        metrics_dictionnary[prop] = scores

    return metrics_dictionnary


def print_grad(module, grad_input, grad_output):
    print('Inside the hook')
    # print('Gradient Input:', grad_input)
    # print('Gradient Output:', grad_output)
    grad_input = grad_input[0]
    grad_output = grad_output[0]

    # difference = torch.norm(grad_input - grad_output)
    # print(f'Gradient Difference: {difference.item()}')


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


def initialize_model_with_lora(args):
    """
    Initialize a VisionEncoderDecoder model and update it with LoRA layers based on given arguments.

    Parameters:

    Returns:
        model (VisionEncoderDecoder): The updated VisionEncoderDecoder model.
    """
    print("Model initialized")
    model = VisionEncoderDecoder(model_type=args.model_type)

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
        r=int(args.lora_rank),
        lora_alpha=2 * int(args.lora_rank),  # 8 normally
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules
    )

    model.main_model = get_peft_model(model.main_model, config)
    model.main_model.print_trainable_parameters()

    dir_adapter = args.ckpt
    if dir_adapter is not None:
        checkpoint_name = f"{dir_adapter}/adapter_model.bin"
        adapters_weights = torch.load(checkpoint_name)
        try:
            cross_attention_layers = f"{dir_adapter}/custom_layers.pth"
            adapters_weights_additionnal = torch.load(cross_attention_layers)
            model.main_model.load_state_dict(adapters_weights_additionnal, strict=False)
            print("loaded")
        except Exception as e:
            print(e)
        print(f"peft weights loaded from {checkpoint_name}")
        set_peft_model_state_dict(model.main_model, adapters_weights)
    print("MODEL INITIALIZED")
    return model


def main_clevr(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = initialize_model_with_lora(args=args)
    dataset_semantic_change = []
    dataset_no_change = []
    modules_to_save = []
    best_score_path = "your_DIR/mono_magic/best_Cider_score.txt"

    concat_mode = args.concat_mode
    if args.dataset == "clevr":
        img_dir = "your_DIR/clevr_dataset/data/data/images"
        sc_dir = "your_DIR/clevr_dataset/data/data/sc_images"
        nsc_dir = "your_DIR/clevr_dataset/data/data/nsc_images"

        dataset_semantic_change = ClevrChangeDataset(img_dir=img_dir, modified_img_dir=sc_dir,
                                                     processor=model.processor,
                                                     model_type="opt", data_pair="modified")
        dataset_no_change = ClevrChangeDataset(img_dir=img_dir, modified_img_dir=nsc_dir,
                                               processor=model.processor,
                                               model_type="opt", data_pair="unchanged")

    if args.dataset == "spot":
        print("DATASET : SPOT")
        img_dir = "your_DIR/spot_the_diff/resized_images/original"
        sc_dir = "your_DIR/spot_the_diff/resized_images/modified"

        print("ARCHITECTURE : mono")

        train_set = SpotTheDiff(img_dir=img_dir, modified_img_dir=sc_dir, processor=model.processor,
                                label_file="your_DIR/spot_the_diff/reformat_train.json")
        val_set = SpotTheDiff(img_dir=img_dir, modified_img_dir=sc_dir, processor=model.processor,
                              label_file="your_DIR/spot_the_diff/reformat_val.json")
        test_set = SpotTheDiff(img_dir=img_dir, modified_img_dir=sc_dir, processor=model.processor,
                               label_file="your_DIR/spot_the_diff/reformat_test.json")

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

        img_dir = "your_DIR/IER_dataset/images"
        img_dir_synthetic = "your_DIR/IER_filtered_top_two_above_threshold_0.3"

        train_data_original = ImageEditingRequest(img_dir=img_dir, img_dir_synthetic=img_dir,
                                                  processor=model.processor,
                                                  label_file="your_DIR/IER_dataset/train.json",
                                                  concat_mode=concat_mode)

        train_data_synthetic = ImageEditingRequest(img_dir=img_dir, img_dir_synthetic=img_dir_synthetic,
                                                   processor=model.processor,
                                                   label_file="your_DIR/IER_filtered_top_two_above_threshold_0.3/synthetic_train.json",
                                                   concat_mode=concat_mode)
        combined_train_data = ConcatDataset([train_data_original, train_data_synthetic])

        train_loader = DataLoader(combined_train_data, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                  num_workers=32,
                                  pin_memory=True, drop_last=True)
        test_data = ImageEditingRequest(img_dir=img_dir, img_dir_synthetic=None, processor=model.processor,
                                        label_file="your_DIR/IER_dataset/test.json",
                                        concat_mode=concat_mode)

        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                 num_workers=32,
                                 pin_memory=True, drop_last=False)

    if args.dataset == "clevr":
        print("clevr dataset used")
        if dataset_no_change != []:
            concatenated_dataset = ConcatDataset([dataset_semantic_change, dataset_no_change])
        else:
            concatenated_dataset = dataset_semantic_change
        with open("your_DIR/clevr_dataset/data/data/splits.json", "r") as f:
            split_info = json.load(f)

        train_idx = split_info['train']
        train_nc_idx = [x + len(dataset_semantic_change) for x in train_idx]
        train_data = Subset(concatenated_dataset, train_idx)
        train_nc_data = Subset(concatenated_dataset, train_nc_idx)
        train_dataset = ConcatDataset([train_data, train_nc_data])

        train_all_idx = train_idx + train_nc_idx

        five_percent_length = int(0.25 * len(train_dataset))
        indices = torch.randperm(len(train_dataset))[:five_percent_length]
        # indices = [x for x in range(len(train_dataset))][:five_percent_length]
        # Create a new Subset using the random indices
        subset_dataset = Subset(train_dataset, indices)

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
       
    if args.dataset == "DC":
        clevr_dc_dataset = ClevrDC_Dataset(processor=model.processor)

        split_json = "your_DIR/BIG_storage/clevr_dc/split_dc.json"
        with open(split_json, 'r') as file:
            split_info = json.load(file)

        train_idx = split_info['train']
        train_data = Subset(clevr_dc_dataset, train_idx)

        test_idx = split_info['test']#[0:320]
        test_data = Subset(clevr_dc_dataset, test_idx)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=custom_collate_clevr_dc, shuffle=True,
                                num_workers=32,
                                pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=custom_collate_clevr_dc, shuffle=False,
                                num_workers=32,
                                pin_memory=True, drop_last=True)
    if args.dataset == "magic":
        # On this section we finetune on MagicBrsuh to evaluate on spot the diff
        img_dir = "your_DIR/IER_dataset/images"
        dataset = load_dataset("osunlp/MagicBrush")


        train_data = MagicDataset(dataset["train"], model.processor, concat_mode=concat_mode)
        val_data = MagicDataset(dataset["dev"], model.processor, concat_mode=concat_mode)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                  num_workers=32,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                num_workers=32,
                                pin_memory=True, drop_last=True)

    if args.dataset == "emu":
        def custom_collate_emu(batch, processor=model.processor):
            # Define your custom collate function here

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
                double_img.save("double_image.jpg")
                task_prompt = "Describe the differences between the two images:"
                inputs = processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
                    0).half()
                double_images.append(inputs)

                idxs.append(sample["idx"])
                tasks.append(sample['task'])
                instructions.append(sample['instruction'])

            return torch.stack(double_images), instructions, tasks, idxs

        dataset = load_dataset("facebook/emu_edit_test_set_generations",
                               cache_dir="your_DIR/.cache/huggingface/datasets")
        data = dataset["validation"]
        og_train_data = []
        og_val_data = []
        
        split_file="your_DIR/emu_dataset/splits.json"
        with open(split_file,"r") as file :
            split_data = json.load(file)
        
        for x in data: 
            if x["idx"] in split_data["train"]:
                og_train_data.append(x)
            if x["idx"] in split_data["validation"]:
                og_val_data.append(x)
        test_data = dataset["test"]
        train_data = EmuDataset(processor=model.processor,split="train")
        val_data = EmuDataset(processor=model.processor,split="validation")
        select_rate = args.select_rate
        #sampler = FlexibleSampler(dataset_emu, select_rate=select_rate)
        
        if args.synth_pretraining == "True": #means pretraining has been done on customized synthetic data before
            print("use ckpt : ", args.synth_pretraining)

            # Create a data loader for the combined dataset
            train_loader = DataLoader(og_train_data,batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                      shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
            val_loader   = DataLoader(og_val_data,batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                      shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
        else: # you train on your synthetic augmentation data first
            # Create a data loader for the combined dataset
            train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                       num_workers=32, pin_memory=True, drop_last=True)
            val_loader = DataLoader(og_val_data, batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                       num_workers=32, pin_memory=True, drop_last=True)#WE USE THE SAME VAL SPLIT AS EMU OG
            # train_loader = DataLoader(combined_dataset, batch_size=16, collate_fn=custom_collate_emu,
            #                           shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
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
    logger = get_logger("results/log.txt")
    best_metrics = []
    try:
        with open("your_DIR/all_run_scores.json", 'r') as file:
            run_scores = json.load(file)
    except FileNotFoundError:
        run_scores = {}
        with open("your_DIR/all_run_scores.json", 'w') as file:
            json.dump(run_scores, file, indent=4)
    if args.little_name not in run_scores.keys():
        run_scores[args.little_name] = {}
        run_scores[args.little_name]["best_cider_score"] = 0.0
        run_scores[args.little_name][
            "path"] = f"your_DIR/{args.save_dir}/{args.little_name}_{args.lora_rank}.bin.opt"

    if args.architecture == "mono":
        best_loss = 10.0
        if args.TRAIN == "True":
            print("Beginning training with validation")
            val_loss = []
            for epoch in tqdm(range(num_epochs), total=num_epochs, desc=f'epoch {epochs}'):
                train(model, train_loader, optimizer, device, args)

                if args.dataset == "magic":
                    val_loss = 1
                    CIDEr, metrics = eval_epoch(model.main_model, dataloader=val_loader, device=device, epoch=epoch,
                                                logger=logger,
                                                processor=model.processor, args=args)
                else:
                    avg_loss = validation(model, dataloader=val_loader, device=device, args=args)
                    val_loss.append(avg_loss)
                print(val_loss)
                
           
                if avg_loss < best_loss:
                    early_stop = 0
                    best_loss = avg_loss
                    output_model_file = save_model(epoch, model, save_path=run_scores[args.little_name]["path"],
                                                   logger=logger)
                    
                    best_output_model_file = output_model_file
                    
                    with open("your_DIR/all_run_scores.json", 'w') as file:
                        json.dump(run_scores, file, indent=4)
                    
                    logger.info(
                        "The best model is: {}".format(best_output_model_file))
                else : 
                    early_stop += 1
                   

                if early_stop == 10:
                    break        

        if args.TEST == "True":
            if args.TRAIN == "True":
                args.ckpt = run_scores[args.little_name]["path"]
                print(args.ckpt)
            model = initialize_model_with_lora(args=args)
            if args.dataset in ["emu","IER"] :
                # print("testing task emu")
                # CIDEr, task_metrics = eval_emu_mono(model.main_model, task_dataloaders=task_dataloaders, device=device,
                #                                processor=model.processor,
                #                                args=args)  # , mean_sentence="there is a person walking in the parking lot")
                # logger.info(f"test scores are {task_metrics}, best CIDEr is {CIDEr}")
                print("TESTING overall Emu")
                CIDEr, metrics = eval_epoch(model.main_model, dataloader=test_loader, device=device, epoch="eval",
                                            logger=logger,
                                            processor=model.processor,
                                            args=args)  # , mean_sentence="there is a person walking in the parking lot")
                run_scores[args.little_name]["best_cider_score"] = CIDEr
                print(metrics)

                with open("your_DIR/all_run_scores.json", 'w') as file:
                    json.dump(run_scores, file, indent=4)
                
                logger.info(f"test scores are {metrics}, best CIDEr is {CIDEr}")
                print("Tested")
            
        if args.TEST == "True":
            print("TESTING")
            CIDEr, metrics = eval_epoch(model.main_model, dataloader=test_loader, device=device, epoch="eval",
                                        logger=logger,
                                        processor=model.processor, args=args,
                                        mean_sentence="the scene remains the same")
            logger.info(f"test scores are {metrics}, best CIDEr is {CIDEr}")
            print("Tested")
    print("END TRAINING")
    return best_score, best_metrics


def main_dataset():
    img_dir = "your_DIR/clevr_dataset/data/data/images"
    sc_dir = "your_DIR/clevr_dataset/data/data/sc_images"
    nsc_dir = "your_DIR/clevr_dataset/data/data/nsc_images"

    model = VisionEncoderDecoder(model_type="opt")

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        # target_modules=["qkv"]
        target_modules=[f"qformer.encoder.layer.{x}.crossattention.attention.query" for x in range(0, 12)] + ["qkv",
                                                                                                              "value",
                                                                                                              "key",
                                                                                                              "q_proj",
                                                                                                              "k_proj",
                                                                                                              "v_proj"]
    )
    model.main_model = get_peft_model(model.main_model, config)
    model.main_model.print_trainable_parameters()
    dir_adapter = "FullFT/thirdborn.bin.opt.3"
    if dir_adapter is not None:
        checkpoint_name = f"{dir_adapter}/adapter_model.bin"
        adapters_weights = torch.load(checkpoint_name)

        print(f"peft weights loaded from {checkpoint_name}")
        set_peft_model_state_dict(model.main_model, adapters_weights)

    dataset_semantic_change = ClevrChangeDatasetMono(img_dir=img_dir, modified_img_dir=sc_dir,
                                                     processor=model.processor,
                                                     model_type="opt", data_pair="modified")
    dataset_no_change = ClevrChangeDatasetMono(img_dir=img_dir, modified_img_dir=nsc_dir,
                                               processor=model.processor,
                                               model_type="opt", data_pair="unchanged")

    concatenated_dataset = ConcatDataset([dataset_semantic_change, dataset_no_change])
    with open("your_DIR/clevr_dataset/data/data/splits.json", "r") as f:
        split_info = json.load(f)

    train_idx = split_info['train']
    val_idx = split_info['val']
    test_idx = split_info['test']  # only accounting for the first dataset
    test_nc_idx = [x + len(dataset_semantic_change) for x in test_idx]
    test_data = Subset(concatenated_dataset, test_idx)
    test_nc_data = Subset(concatenated_dataset, test_nc_idx)
    test_dataset = ConcatDataset([test_data, test_nc_data])
    # print(test_dataset[0],test_dataset[0+len(test_data)])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False, num_workers=32,
                             pin_memory=True)
    eval_epoch(model.main_model, test_loader, epoch=0, device='cuda', processor=model.processor)


def main_spot():
    print("STARTING")
    model = VisionEncoderDecoder(model_type="opt")

    img_dir = "your_DIR/spot_the_diff/resized_images/original"
    sc_dir = "your_DIR/spot_the_diff/resized_images/modified"

    if args.architecture == "mono":
        dataset_semantic_change = SpotTheDiff(img_dir=img_dir, modified_img_dir=sc_dir, processor=model.processor,
                                              label_file="your_DIR/spot_the_diff/reformat_train.json")
    set_loader = DataLoader(dataset_semantic_change, batch_size=32, collate_fn=custom_collate, shuffle=False,
                            num_workers=32,
                            pin_memory=True, drop_last=True)


def main_prop():
    computing_properties()
    print("testing properties done")


def main_magic():
    print("using main_magic")
    model = VisionEncoderDecoder(model_type="opt")
    dataset = load_dataset("osunlp/MagicBrush")
    train_data = MagicDataset(dataset["train"], model.processor)
    train_loader = DataLoader(train_data, batch_size=32, collate_fn=custom_collate, shuffle=False,
                              num_workers=32,
                              pin_memory=True, drop_last=True)
    for i in train_loader:
        print(i)


def main_IER():
    print("using main_magic")
    model = VisionEncoderDecoder(model_type="opt")
    img_dir = "your_DIR/IER_dataset/images"
    train_data = ImageEditingRequest(img_dir=img_dir, processor=model.processor,
                                     label_file="your_DIR/IER_dataset/train.json")
    train_loader = DataLoader(train_data, batch_size=32, collate_fn=custom_collate_emu, shuffle=False,
                              num_workers=32,
                              pin_memory=True, drop_last=True)
    for i in train_loader:
        print(i)


def main_emu():
    print("using emu")
    model = VisionEncoderDecoder(model_type="opt")

    def custom_collate_emu(batch, processor=model.processor):
        # Define your custom collate function here

        # Initialize lists to store processed data
        double_images = []
        tasks = []
        instructions = []
        idxs = []
        for sample in batch:
            # Load the PNG image using PIL (Pillow)
            image = sample['image'].convert('RGB')
            edited_image = sample['edited_image'].convert('RGB')
            # Save the images to your desired location
            idx = sample['idx']
            double_img = Image.new('RGB', (image.width, image.height + edited_image.height))
            double_img.paste(image, (0, 0))
            double_img.paste(edited_image, (0, image.height))
            idxs.append(idx)
            double_img = composed_transforms(double_img)

            task_prompt = "Describe the differences between the two images:"
            inputs = processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
                0).half()
            double_images.append(inputs)
            tasks.append(sample['task'])
            instructions.append(sample['instruction'])

        return torch.stack(double_images), instructions, tasks, idxs

    dataset = load_dataset("facebook/emu_edit_test_set_generations")
    data = dataset["validation"]
    train_data = EmuDataset(processor=model.processor,split="train")
    
    # Use the filter method to select only those examples

    
    loader = DataLoader(train_data, batch_size=32,
                        shuffle=True, num_workers=32, collate_fn=custom_collate_emu, pin_memory=True, drop_last=True)
    for batch in loader:
        print(batch)
    print(len(loader))
    print("loaded")
    exit()
    loader = DataLoader(data, batch_size=args.batch_size,
                        shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    dict_captions = {}
    for batch in tqdm(data, total=len(loader)):
        index = batch["idx"]
        if index not in dict_captions:
            dict_captions[index] = {}
        dict_captions[index]["input_caption"] = batch["input_caption"]
        dict_captions[index]["output_caption"] = batch["output_caption"]
    with open('captions.json', 'w') as file:
        json.dump(dict_captions, file)
    print("saved")
    exit()
    from collections import defaultdict

    # Load the dataset
    # Initialize a dictionary to hold subsets
    task_subsets = defaultdict(list)

    # Iterate through the dataset and populate subsets
    # for item in dataset['test']:
    #     task_subsets[item['task']].append(item)

    # Function to create a DataLoader for a given subset
    def create_dataloader_for_task(task_data, collate_fn=custom_collate_emu, batch_size=32):
        return DataLoader(task_data, batch_size=batch_size, collate_fn=collate_fn,
                          shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

    # Create a DataLoader for each task
    task_dataloaders = {task: create_dataloader_for_task(data, custom_collate_emu) for task, data in
                        task_subsets.items()}
    # print(task_dataloaders)
    for task, dataloader in task_dataloaders.items():
        for batch in dataloader:
            print(batch)


def main_mean_sentence():
    model = VisionEncoderDecoder(model_type="opt")
    if False:
        img_dir = "your_DIR/clevr_dataset/data/data/images"
        sc_dir = "your_DIR/clevr_dataset/data/data/sc_images"
        nsc_dir = "your_DIR/clevr_dataset/data/data/nsc_images"


        dataset_semantic_change = ClevrChangeDataset(img_dir=img_dir, modified_img_dir=sc_dir,
                                                     processor=model.processor,
                                                     model_type="opt", data_pair="modified")
        dataset_no_change = ClevrChangeDataset(img_dir=img_dir, modified_img_dir=nsc_dir,
                                               processor=model.processor,
                                               model_type="opt", data_pair="unchanged")
        if dataset_no_change != []:
            concatenated_dataset = ConcatDataset([dataset_semantic_change, dataset_no_change])
        else:
            concatenated_dataset = dataset_semantic_change
        with open("your_DIR/clevr_dataset/data/data/splits.json", "r") as f:
            split_info = json.load(f)

        train_idx = split_info['train']
        train_nc_idx = [x + len(dataset_semantic_change) for x in train_idx]
        train_data = Subset(concatenated_dataset, train_idx)
        train_nc_data = Subset(concatenated_dataset, train_nc_idx)
        train_dataset = ConcatDataset([train_data, train_nc_data])

        set_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False,
                                num_workers=32,
                                pin_memory=True, drop_last=True)
    if True:
        def custom_collate_emu(batch, processor=model.processor):
            # Define your custom collate function here

            # Initialize lists to store processed data
            double_images = []
            tasks = []
            instructions = []
            idxs = []
            for sample in batch:
                image = sample['image'].convert('RGB')
                edited_image = sample['edited_image'].convert('RGB')
                original_size = edited_image.size
                new_image = image.resize(original_size, Image.BICUBIC)
                idx = sample["idx"]
                double_img = Image.new('RGB', (new_image.width, new_image.height + edited_image.height))
                double_img.paste(new_image, (0, 0))
                double_img.paste(edited_image, (0, new_image.height))

                double_img = composed_transforms(double_img)
                double_img.save("double_image.jpg")
                task_prompt = "Describe the differences between the two images:"
                inputs = processor(images=double_img, text=task_prompt, return_tensors="pt")["pixel_values"].squeeze(
                    0).half()
                double_images.append(inputs)
                tasks.append(sample['task'])
                instructions.append(sample['instruction'])

            return torch.stack(double_images), instructions, tasks, idxs

        dataset = load_dataset("facebook/emu_edit_test_set_generations",
                               cache_dir="your_DIR/.cache/huggingface/datasets")
        data = dataset["validation"]
        test_data = dataset["test"]
        dataset_emu = EmuDataset(processor=model.processor)

        # Create a data loader for the combined dataset
        train_loader = DataLoader(dataset_emu, batch_size=args.batch_size, collate_fn=custom_collate_emu,
                                  shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
        # Create a data loader for the combined dataset
    corpus = []
    for item in tqdm(train_loader, total=len(train_loader)):
        inputs, labels, idi = item
        corpus.append(labels[0])
    find_mean_sentence_semantic(corpus)


import torch


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


def testing_clevrDC():
    model = VisionEncoderDecoder(model_type="opt")
    clevr_dc_dataset = ClevrDC_Dataset(processor=model.processor)


    split_json = "your_DIR/BIG_storage/clevr_dc/split_dc.json"
    with open(split_json, 'r') as file:
        split_info = json.load(file)

    train_idx = split_info['train']
    train_data = Subset(clevr_dc_dataset, train_idx)
    print(len(train_data))


    set_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=custom_collate_clevr_dc, shuffle=False,
                            num_workers=32,
                            pin_memory=True, drop_last=True)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Use the 'spawn' start method
    # wandb.finish()
    # # start a new wandb run to track this script
    # wandb.init(
    #     project="clevr-change", name="batchsize_1__lr_1E-4"
    # )
    seed = 42  #some time it was 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("WITH SEED 42")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Vision Encoder-Decoder Model Selection")
    parser.add_argument('--model_type', type=str, default='opt',
                        help='Type of model to use. Options are: "opt")
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
    parser.add_argument("--select_rate", type=int, default=1,
                        help='used for sampling to check synthetic consistency')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='select the learning rate for finetuning, between 1e-4 and 1e-5')
    parser.add_argument("--module_to_ft", type=str, default=['ViT', 'QFormer', 'LLM'],
                        help=" which module to finetune using LoRA ? available : ViT, QFormer, LLM")
    parser.add_argument("--vit_embeddings", type=str, default=False,
                        help=" whether to extract the embeddings from the ViT")
    parser.add_argument("--dataset", type=str,
                        help="clevr, spot, IER, Emu")
    parser.add_argument("--synth_pretraining", type=str, default=False,
                        help="do your pretrain is based on synthetic data ?")
    parser.add_argument("--concat_mode", type=str, default="vertical",
                        help="vertical or horizontal concatenation of inputs")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    # main_spot()
    # print("doing mean sentence")
    # main_mean_sentence()
    # exit()
    lora_influence = False
    validation_augmentation = False
    only_classic = False
    # main_IER()
    CIDEr_list = []
    metrics_list = []
    lora_ranks = args.lora_rank
    if lora_influence:
        print("COMPUTING LORA INFLUENCE")
        if type(lora_ranks) == "list":
            for i in range(len(lora_ranks)):
                args.lora_rank = lora_ranks[i]
                best_CIDEr, best_metrics = main_clevr(args)
                CIDEr_list.append(best_CIDEr)
                metrics_list.append(best_metrics)
                with open(f'/xp_results/lora_influence/CIDEr_lora_{args.lora_rank}.txt', 'w') as f:
                    for item in CIDEr_list:
                        f.write("%s\n" % item)

                # Save to JSON file
                with open(f'/xp_results/lora_influence/metrics_list_lora_{args.lora_rank}.json', 'w') as f:
                    json.dump(metrics_list, f)
    print(f"val_aug : {validation_augmentation}, only classic augmentation : {only_classic}")
    if validation_augmentation:
        print("computing with classic and MB transforms")
        for i in range(30):
            print("ITERATION ", i)
            best_CIDEr, best_metrics = main_clevr(args)
            CIDEr_list.append(best_CIDEr)
            metrics_list.append(best_metrics)

            if only_classic:
                with open(f'augmentation_result/classic_augment_emu.txt', 'w') as f:
                    for item in CIDEr_list:
                        f.write("%s\n" % item)

                # Save to JSON file
                with open(f'augmentation_result/classic_augment_emu.json', 'w') as f:
                    json.dump(metrics_list, f)

            else:
                with open(f'augmentation_result/{args.little_name}.txt', 'w') as f:
                    for item in CIDEr_list:
                        f.write("%s\n" % item)

                # Save to JSON file
                with open(f'augmentation_result/{args.little_name}.json', 'w') as f:
                    json.dump(metrics_list, f)
    else:
        #main_emu()
        _, _ = main_clevr(args)
        # main_mean_sentence()
        # def profile_main_clevr():
        #     _, _ = main_clevr(args)

        # Run the profiler
        # cProfile.run('profile_main_clevr()', 'profile_results')
