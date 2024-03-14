# Details

We will release the Syned dataset upon publication. 

ground_truth_variations.py is the script used to generate the variations of the ground truth using Llama-2-7b-chat-hf.

Run this command to generate variations. You can chose to only augment the test set or the train set (that we split into train/val while training).
``` bash
python ground_truth_variations.py --split test
```

