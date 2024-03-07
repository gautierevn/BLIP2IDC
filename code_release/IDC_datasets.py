import os
import json
from torch.utils.data import Dataset
from PIL import Image
import transformers
from torchvision.transforms import ToTensor



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
    def __init__(self, processor, json_file="your_DIR/clevr_dc/captions_dc.json",
                 concat_mode='vertical', data_path="your_DIR/clevr_dc"):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.to_tensor = ToTensor()
        self.processor = processor
        self.concat_mode = concat_mode 
        self.data_path = data_path
        self.id = list(self.data.keys())
        
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

class EmuDataset(Dataset):
    def __init__(self, processor, split="train",json_file="your_DIR/emu_dataset/augmented_dataset.json", split_file="your_DIR/emu_dataset/splits.json", 
                 concat_mode='vertical', use_distractors=False):
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        source_img_path = sample['image']
        target_img_path = sample['edited_image']

        img_id = sample['idx']

        source_img = Image.open(source_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')
        
        sample["image"] = source_img
        sample["edited_image"] = target_img
        return sample