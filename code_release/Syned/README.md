# Details

The Syned dataset is available on the Hugging Face Hub: [Gevennou/Syned](https://huggingface.co/datasets/Gevennou/Syned).

Download it with:

```bash
hf download Gevennou/Syned --repo-type=dataset --local-dir Syned
```

or from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Gevennou/Syned",
    repo_type="dataset",
    local_dir="Syned",
)
```

ground_truth_variations.py is the script used to generate the variations of the ground truth using Llama-2-7b-chat-hf.
Run this command to generate variations. You can chose to only augment the test set or the train set (that we split into train/val while training).

```bash
python ground_truth_variations.py --split test
```
