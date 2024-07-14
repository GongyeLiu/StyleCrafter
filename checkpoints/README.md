# Setup Instructions for StyleCrafter


To set up the environment for StyleCrafter, download the following 3 models and place them in the corresponding paths.

## Step 1: Download OpenCLIP

Navigate to the `path/to/StyleCrafter` directory and download OpenCLIP:

```bash
cd ./checkpoints/open_clip
git lfs install
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

## Step 2: Download VideoCrafter Checkpoint
Download VideoCrafter checkpoint from [huggingface](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt), and put the `model.ckpt` file in the `./checkpoints/videocrafter_t2v_320_512/`;


## Step 3: Download StyleCrafter Checkpoint
Download StyleCrafter checkpoint from [huggingface](https://huggingface.co/liuhuohuo/StyleCrafter/tree/main), put the `adapter_v1.pth` and `temporal_v1_1.pth` files in the `./checkpoints/stylecrafter/`


# Final Directory Structure

After completing the setup, your directory structure should look like this. This structure helps ensure that the models are correctly loaded and validated:


```
VideoCrafter
├── checkpoints
│   ├── open_clip
│   │   └── CLIP-ViT-H-14-laion2B-s32B-b79K
│   │       ├── config.json
│   │       ├── merges.txt
│   │       ├── open_clip_config.json
│   │       ├── open_clip_pytorch_model.bin
│   │       ├── preprocessor_config.json
│   │       ├── pytorch_model.bin
│   │       ├── README.md
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       ├── tokenizer.json
│   │       └── vocab.json
│   ├── README.md
│   ├── stylecrafter
│   │   ├── adapter_v1.pth
│   │   └── temporal_v1_1.pth
│   └── videocrafter_t2v_320_512
│       └── model.ckpt
├── configs
├── ...
```
