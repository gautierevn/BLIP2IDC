import torch
from datasets import load_dataset

from PIL import Image
from torchvision.transforms import ToTensor, transforms
import numpy as np
from transformers import Blip2ForConditionalGeneration

from gradio.components import Image, Dropdown

try:
    from transformers import Blip2ForConditionalGeneration2Images
except ImportError:
    pass
import numpy as np
import torch
from PIL import Image
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from torch import nn
from torchvision.transforms import ToTensor, transforms
from transformers import Blip2Processor, AutoTokenizer

global logger
from torchvision import transforms
import gradio as gr


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


def initialize_model_with_lora(module_to_ft=['ViT', 'QFormer', 'LLM']):
    """
    Initialize a VisionEncoderDecoder model and update it with LoRA layers based on given arguments.

    Parameters:

    Returns:
        model (VisionEncoderDecoder): The updated VisionEncoderDecoder model.
    """
    print("Mono Model initialized")
    model = VisionEncoderDecoder(model_type="opt")

    modules = module_to_ft
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

    config = LoraConfig(
        r=8,
        lora_alpha=2 * int(8),  # 8 normally
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules
    )

    model.main_model = get_peft_model(model.main_model, config)
    model.main_model.print_trainable_parameters()

    possible_ckpt = ["/your_dir/mono_magic/ft_on_emu_classic_augment_8.bin.opt",
                     "/your_dir/mono_magic/magic_training_8.bin.opt"]
    dir_adapter = possible_ckpt[0]
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


def process_for_display(image_tensor):
    """
    Convert a PyTorch tensor to a PIL image for display.

    Args:
    image_tensor (torch.Tensor): The image tensor to convert.

    Returns:
    PIL.Image: The converted PIL image.
    """
    # Check if the tensor is on a CUDA device and move it to CPU
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    # The normalization mean and std used during your initial image preprocessing
    # Adjust these values based on your specific preprocessing pipeline
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Unnormalize the image
    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    image_tensor = unnormalize(image_tensor)

    # Convert from tensor format (C x H x W) to PIL image
    image_pil = transforms.ToPILImage()(image_tensor)

    return image_pil


def generate_caption(model, image1, image2):
    def custom_collate_emu(batch, processor=model.processor):
        # Initialize lists to store processed data
        processed_images = []

        for image1, image2 in batch:
            # Process each pair of images
            image1 = Image.fromarray(np.uint8(image1)).convert('RGB')
            image2 = Image.fromarray(np.uint8(image2)).convert('RGB')

            # Determine the new size for both images (for example, the size of the smaller image)
            new_size = (min(image1.width, image2.width), min(image1.height, image2.height))

            # Resize images using the Lanczos filter
            image1 = image1.resize(new_size, Image.Resampling.LANCZOS)
            image2 = image2.resize(new_size, Image.Resampling.LANCZOS)

            double_img = Image.new('RGB', (max(image1.width, image2.width), image1.height + image2.height))
            double_img.paste(image1, (0, 0))
            double_img.paste(image2, (0, image1.height))

            # Prepare for the model
            inputs = processor(images=double_img, return_tensors="pt")["pixel_values"].squeeze(0).half()
            processed_images.append(inputs)

        # Stack and return the processed images
        return torch.stack(processed_images)

    # Process the images
    processed_images = custom_collate_emu([(image1, image2)])

    # Generate caption
    generated_ids = model.main_model.generate(processed_images.cuda(), max_new_tokens=77)
    generated_text = model.processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text


def main(args):
    # Initialize the model
    model_lora = initialize_model_with_lora()

    # Define the Gradio interface function
    def gradio_interface(image1, image2, model=model_lora):
        caption = generate_caption(model, image1, image2)
        return caption

    # Set up the Gradio interface
    demo = gr.Interface(
        fn=gradio_interface,
        inputs=[gr.Image(), gr.Image()],
        outputs=gr.Textbox(),
        title="Image Difference Captioning Demo",
        description="Upload two images to generate a caption describing the difference from top to bottom."
    )

    # Launch the Gradio interface
    demo.launch(share=True, enable_queue=False)


if __name__ == "__main__":
    
    main(args)
