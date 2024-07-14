# StyleCrafter: Enhancing Stylized Text-to-Video Generation with Style Adapter

### 🔥🔥🔥 [StyleCrafter on SDXL](https://github.com/GongyeLiu/StyleCrafter-SDXL) for stylized image generation is available! Enabling higher resolution(1024×1024) and more visually pleasing!


<div align="center">

 <a href='https://arxiv.org/abs/2312.00330'><img src='https://img.shields.io/badge/arXiv-2312.00330-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://gongyeliu.github.io/StyleCrafter.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://huggingface.co/spaces/liuhuohuo/StyleCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
 <a href='https://github.com/GongyeLiu/StyleCrafter'><img src='https://img.shields.io/badge/StyleCrafter-VideoCrafter-darkcyan'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href='https://github.com/GongyeLiu/StyleCrafter-SDXL'><img src='https://img.shields.io/badge/StyleCrafter-SDXL-darkcyan'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


_**[GongyeLiu](https://github.com/GongyeLiu), [Menghan Xia*](https://menghanxia.github.io/), [Yong Zhang](https://yzhang2016.github.io), [Haoxin Chen](https://scholar.google.com/citations?user=6UPJSvwAAAAJ&hl=zh-CN&oi=ao), [Jinbo Xing](https://doubiiu.github.io/), <br>[Xintao Wang](https://xinntao.github.io/), [Yujiu Yang*](https://scholar.google.com/citations?user=4gH3sxsAAAAJ&hl=zh-CN&oi=ao), [Ying Shan](https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate)**_
<br><br>
(* corresponding authors)

From Tsinghua University and Tencent AI Lab.

</div>


## 🔆 Introduction

**TL;DR:** We propose StyleCrafter, a generic method that enhances pre-trained T2V models with style control, supporting Style-Guided Text-to-Image Generation and Style-Guided Text-to-Video Generation.  <br>


### 1. ⭐⭐ Style-Guided Text-to-Video Generation.

<div align="center">
<img src=docs/showcase_1.gif>
<p>Style-guided text-to-video results. Resolution: 320 x 512; Frames: 16. (Compressed)</p>
</div>


### 2. Style-Guided Text-to-Image Generation.

<div align="center">
<img src=docs/showcase_img.jpeg>
<p>Style-guided text-to-image results. Resolution: 512 x 512. (Compressed)</p>
</div>


## 📝 Changelog
- __[2024.07.14]__: 🔥🔥 Remove watermarks! We finetune temporal blocks on non-watermark data with 500 steps, download updated checkpoints [here](https://huggingface.co/liuhuohuo/StyleCrafter/blob/main/temporal_v1_1.pth).
- __[2024.06.25]__: 🔥🔥 Support [StyleCrafter on SDXL](https://github.com/GongyeLiu/StyleCrafter-SDXL)!
- __[2023.12.08]__: 🔥🔥 Release the Huggingface online [demo](https://huggingface.co/spaces/liuhuohuo/StyleCrafter).
- __[2023.12.05]__: 🔥🔥 Release the code and checkpoint.
- __[2023.11.30]__: 🔥🔥 Release the project page.



## 🧰 Models

|Base Model| Gen Type | Resolution | Checkpoint | How to run |
|:---------|:---------|:--------|:--------|:--------|
|VideoCrafter| Image/Video |320x512|[Hugging Face](https://huggingface.co/liuhuohuo/StyleCrafter/tree/main)| [StyleCrafter on VideoCrafter](https://github.com/GongyeLiu/StyleCrafter)
|SDXL| Image |1024x1024|[Hugging Face](https://huggingface.co/liuhuohuo/StyleCrafter-SDXL)| [StyleCrafter on SDXL](https://github.com/GongyeLiu/StyleCrafter-SDXL)


It takes approximately 5 seconds to generate a 512×512 image and 85 seconds to generate a 320×512 video with 16 frames using a single NVIDIA A100 (40G) GPU. A GPU with at least 16G GPU memory is required to perform the inference process.

## ⚙️ Setup

```bash
conda create -n stylecrafter python=3.8.5
conda activate stylecrafter
pip install -r requirements.txt
```

## 💫 Inference

1) Download all checkpoints according to the [instructions](./checkpoints/README.md)
2) Run the commands in terminal.
```bash
# style-guided text-to-image generation
sh scripts/run_infer_image.sh

# style-guided text-to-video generation
sh scripts/run_infer_video.sh
```
3) (Optional) Infernce on your own data according to the [instructions](./eval_data/README.md)



## 👨‍👩‍👧‍👦 Crafter Family
[VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter): Framework for high-quality text-to-video generation.

[ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter): Tuning-free method for high-resolution image/video generation.

[TaleCrafter](https://github.com/AILab-CVC/TaleCrafter): An interactive story visualization tool that supports multiple characters.  

[LongerCrafter](https://github.com/arthur-qiu/LongerCrafter): Tuning-free method for longer high-quality video generation.  

[DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter) Animate open-domain still images to high-quality videos.


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****

## 🙏 Acknowledgements
We would like to thank [AK(@_akhaliq)](https://twitter.com/_akhaliq?lang=en) for the help of setting up online demo.


## 📭 Contact
If your have any comments or questions, feel free to contact [lgy22@mails.tsinghua.edu.cn](lgy22@mails.tsinghua.edu.cn)

## BibTex
```bibtex
@article{liu2023stylecrafter,
  title={StyleCrafter: Enhancing Stylized Text-to-Video Generation with Style Adapter},
  author={Liu, Gongye and Xia, Menghan and Zhang, Yong and Chen, Haoxin and Xing, Jinbo and Wang, Xintao and Yang, Yujiu and Shan, Ying},
  journal={arXiv preprint arXiv:2312.00330},
  year={2023}
}
```
