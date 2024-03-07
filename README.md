# BLIP2IDC: Enhancing Image Difference Captioning with Synthetic Augmentation

## Abstract Summary
The evolution of generative models has significantly advanced the generation of image variations. The Image Difference Captioning (IDC) task, aimed at identifying differences between images, faces challenges with real-world images due to data scarcity and the complexity of capturing nuanced differences. We introduce BLIP2IDC, a framework that adapts image captioning models for IDC and uses synthetic augmentation to enrich IDC datasets, demonstrating superior performance on real-world datasets.

## Getting Started

### Prerequisites
Ensure you have access to the necessary datasets for training and evaluation. For synthetic data augmentation and training with BLIP2IDC, the following are required:
- Python 3.8 or higher
- PyTorch 1.7 or higher
- Access to GPU for efficient training

### Installation
1. Clone the BLIP2IDC repository:

``` bash
Copy code
git clone https://github.com/Gautier29/BLIP2IDC.git
```

Install the required Python packages:
bash
Copy code
cd BLIP2IDC
pip install -r requirements.txt
Usage
To start training BLIP2IDC with synthetic data augmentation, follow these steps:

Ensure you have your datasets properly set up in a directory accessible by the script.

Run the training script with the appropriate parameters:

bash
Copy code
./train_BLIP2IDC.sh
This script will automatically use the synthetic augmentation techniques described in our paper to enhance the performance of the BLIP2IDC model.

Accessing the Syned Dataset
Our Syned dataset, designed specifically for IDC, is available under the supplementary section of our GitHub repository. Follow the instructions provided there to download and integrate it with your training process for improved results.

For more detailed usage and additional options, please refer to the supplementary documentation available in the repository.
