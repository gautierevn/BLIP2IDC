import sys
COCO_PATH = 'your_dir/coco-caption' # i.e. /home/user/code/coco-caption
sys.path.insert(0, COCO_PATH)
from cococaption.pycocoevalcap.evil import COCOEvilCap

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.eval import PTBTokenizer, Bleu, Meteor, Rouge, Cider
import json
import copy
import numpy as np

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

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

def coco_gt_format_save(gt_file, neg=False):
    gt = json.load(open(gt_file, 'r'))
    gt_dict = {}
    info_dict = {
        'contributor': 'dummy',
        'date_created': 'dummy',
        'description': 'dummy',
        'url': 'dummy',
        'version': 'dummy',
        'year': 'dummy'
    }

    gt_dict['info'] = info_dict
    gt_dict['licenses'] = info_dict
    gt_dict['type'] = 'captions'
    gt_dict['images'] = []
    gt_dict['annotations'] = []

    count = 0
    for single_dict in gt:
        for k, v in single_dict.items():
            print(k,v)
            image_id = k.split('_')[-1]
            im = {'filename': image_id, 'id': image_id}
            gt_dict['images'].append(im)
            for c in v:
                annotation = {'caption': c, 'id': count, 'image_id': image_id}
                count += 1
                gt_dict['annotations'].append(annotation)

    json.dump(gt_dict, open(gt_file.split('.json')[0] + '_reformat.json', 'w'))

def coco_gen_format(gen_dict):
    results = []
    for k, v in gen_dict.items():
        results.append({'caption': v, 'image_id': k})
    return results

def coco_gen_format_save(gen_dict, save_path):
    results = coco_gen_format(gen_dict)
    json.dump(results, open(save_path, 'w'))

def merge_gt_files(gt_file1, gt_file2, save_path):
    gt1 = json.load(open(gt_file1, 'r'))
    gt2 = json.load(open(gt_file2, 'r'))
    gt_dict = {}
    info_dict = {
        'contributor': 'dummy',
        'date_created': 'dummy',
        'description': 'dummy',
        'url': 'dummy',
        'version': 'dummy',
        'year': 'dummy'
    }

    gt_dict['info'] = info_dict
    gt_dict['licenses'] = info_dict
    gt_dict['type'] = 'captions'
    gt_dict['images'] = []
    gt_dict['annotations'] = []

    count = 0
    for k, v in gt1.items():
        image_id = k.split('_')[-1]
        im = {'filename': image_id, 'id': image_id}
        gt_dict['images'].append(im)
        for c in v:
            annotation = {'caption': c, 'id': count, 'image_id': image_id}
            count += 1
            gt_dict['annotations'].append(annotation)

    for k, v in gt2.items():
        image_id = k.split('_')[-1] + '_n'
        im = {'filename': image_id, 'id': image_id}
        gt_dict['images'].append(im)
        for c in v:
            annotation = {'caption': c, 'id': count, 'image_id': image_id}
            count += 1
            gt_dict['annotations'].append(annotation)

    json.dump(gt_dict, open(save_path, 'w'))

def score_generation(anno_file, result_file):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)

    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()

    coco_eval.evaluate()
    return copy.deepcopy(coco_eval.eval)

def score_generation_by_type(anno_file, result_file, type_file):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)
    coco_eval = COCOEvalCap(coco, coco_res)

    type_dict = json.load(open(type_file, 'r'))
    results = {}
    for type, image_ids in type_dict.items():
        filtered = set(coco_res.getImgIds()).intersection(set(image_ids))
        coco_eval.params['image_id'] = list(filtered)
        coco_eval.evaluate()
        results[type] = copy.deepcopy(coco_eval.eval)

    return results

def score_generation_with_ids(anno_file, result_file, img_ids):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)

    coco_eval = COCOEvalCap(coco, coco_res)
    filtered = set(coco_res.getImgIds()).intersection(set(img_ids))
    coco_eval.params['image_id'] = list(filtered)

    coco_eval.evaluate()
    return copy.deepcopy(coco_eval.eval)

def score_generation_by_type_with_ids(anno_file, result_file, type_file, img_ids):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)
    coco_eval = COCOEvalCap(coco, coco_res)

    type_dict = json.load(open(type_file, 'r'))
    results = {}
    for type, image_ids in type_dict.items():
        filtered = set(coco_res.getImgIds()).intersection(set(image_ids))
        filtered_twice = filtered.intersection(set(img_ids))
        coco_eval.params['image_id'] = list(filtered_twice)
        coco_eval.evaluate()
        results[type] = copy.deepcopy(coco_eval.eval)

    return results

def pointing(gen_mapping, gt_mapping, type_ids=None):
    pointings = []
    count = 0
    if type_ids:
        type_ids = set([str(int(id.split('.')[0])) for id in type_ids])
    for id, (gen_before, gen_after) in gen_mapping.items():
        if type_ids and id not in type_ids:
            continue
        gt_before, gt_after = gt_mapping[id]
        if gt_before is not None:
            gen_before_flat = gen_before.flatten()
            gt_before_flat = gt_before.flatten()
            p_before = gt_before_flat[np.argmax(gen_before_flat)]
            count += 1
        else:
            p_before = 0.0
        if gt_after is not None:
            gen_after_flat = gen_after.flatten()
            gt_after_flat = gt_after.flatten()
            p_after = gt_after_flat[np.argmax(gen_after_flat)]
            count += 1
        else:
            p_after = 0.0
        p = p_before + p_after
        pointings.append(p)
    m_pointing = sum(pointings) / float(count)
    return m_pointing

def coverage(gen_mapping, gt_mapping, type_ids=None):
    coverages = []
    if type_ids:
        type_ids = set([str(int(id.split('.')[0])) for id in type_ids])
    for id, (gen_before, gen_after) in gen_mapping.items():
        # normalize
        gen_before = gen_before / gen_before.sum()
        gen_after = gen_after / gen_after.sum()
        if type_ids and id not in type_ids:
            continue
        gt_before, gt_after = gt_mapping[id]
        if gt_before is not None:
            s_before = (gt_before * gen_before).sum()
        else:
            s_before = 0.0
        if gt_after is not None:
            s_after = (gt_after * gen_after).sum()
        else:
            s_after = 0.0
        score = (s_before + s_after) / 2.0
        coverages.append(score)
    m_coverage = np.mean(coverages)
    return m_coverage

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
    with open("Syned/instructions_with_idx_test_split.txt", 'r') as file:
        for line in file:
            # Remove any trailing newlines or spaces
            idx, instruction = line.strip().split("<INSTRUCT>")
            original_gt[idx]=instruction

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
                    if args.dataset == "IER" or args.augmentation == "IER" or args.dataset == "emu" or args.dataset == "DC":
                        all_result_lists.append(generated_text[c])
                        if args.dataset !="emu":                             
                            gt_list.append(labels[c])
                        else : 
                            gt_list.append(emu_gt_test[f'{id} ']+[original_gt[f'{id} ']])#clean dataset
                    else:
                        all_result_lists.append({"caption": generated_text[c], "image_id": id})

    if args.dataset == "clevr":
        anno_file = "your_DIR/clevr_dataset/data/data/total_change_captions_reformat.json"


    if args.dataset == "IER" or args.dataset == "emu" or args.dataset == "DC":
        print(f"evaluating {args.dataset}")
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
    with open("your_DIR/Syned/gt_augmented_test_clean.json", 'r') as json_file:
        emu_gt_test = json.load(json_file)
    original_gt={}
    
    with open("Syned/instructions_with_idx_test_split.txt", 'r') as file:
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
    
if __name__ == '__main__':

    anno_path = "/your_dir/IER_dataset/test.json"
    coco_gt_format_save(anno_path)
   