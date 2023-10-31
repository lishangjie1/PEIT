# PEIT 
## overview
The codes for ACL2023 paper: "PEIT: Bridging the Modality Gap with Pre-trained Models for End-to-End Image Translation" [[paper](https://aclanthology.org/2023.acl-long.751/)]
## Download ECOIT Dataset
We have released ECOIT Dataset, you can download it [here](https://pan.baidu.com/s/1UYSjSD6GUrt1tTqwgZlH0g?pwd=1fdg).
## installment
pip install -e ./

python setup.py build_ext --inplace
## Folder structure
Assuming the folder structure is as follows:  
<code>/tmp/data
|—— mt_data (parallel text data)
    |—— train.en
    |—— train.zh
    |—— valid.en
    |—— valid.zh
|—— it_data (Synthetic Data)
    |—— train_images
        |—— 0.jpg
        |—— 1.jpg
        |—— ....
        |—— 10000000.jpg
        |—— text.en
        |—— text.zh
    |—— valid_images
        |—— 0.jpg
        |—— 1.jpg
        |—— ....
        |—— 1000.jpg
        |—— text.en
        |—— text.zh
|—— ft_data (ECOIT)
    |—— train_images
        |—— 0.jpg
        |—— 1.jpg
        |—— ....
        |—— 470000.jpg
        |—— text.en
        |—— text.zh
    |—— valid_images
        |—— 0.jpg
        |—— 1.jpg
        |—— ....
        |—— 2000.jpg
        |—— text.en
        |—— text.zh
    |—— test_images
        |—— 0.jpg
        |—— 1.jpg
        |—— ....
        |—— 1000.jpg
        |—— text.en
        |—— text.zh</code>
    

text.en and text.zh in train_images/valid_images/test_images will record sentences line by line based on the id of images.
## Data Synthesis
- Data Synthesis: <code>python data_synthesis_and_preprocess/data_synthesis/data_synthesis.py</code> 
- You need to provide the paths of ECOIT dataset (We will release it soon, which provides backgrounds), pre-downloaded fonts (you can download fonts from [Google Fonts](https://fonts.google.com/)), mt dataset (provides parallel sentence pairs) to create image translation samples in train_images/valid_images.
## Data processing and Training
### Process:  
Data Cleaning  
BPE training with sentencepiece  
BPE apply  
Binarizing the data with fairseq-preprocess  
### Datasets:  
UN dataset

### Training
- mt_data is used to pre-train pure text machine translation model in stage one
- it_data is used to pre-train image translation model in stage two
- ft_data is the ECOIT dataset and will be used to fine-tune image translation model.

- Please refer to <code>run.sh</code> for the complete training process.
- Please refer to <code>run_generate.sh</code> for the image translation inference.

## Multi-Line Image Translation
- Please set <code>--multi-line --model-height 320 --model-width 480</code> in IT training，the visual encoder of IT model will flatten feature maps from multiple rows into one row.

## Pretrained OCR Initialization
- For CRNN visual encoder, we initialize its parameters with a pretrained ocr model (zh_sim_g2.pth), you can download it from [EasyOCR](https://github.com/JaidedAI/EasyOCR).

# Citation
Please cite as:
``` bibtex
@inproceedings{zhu-etal-2023-peit,
    title = "{PEIT}: Bridging the Modality Gap with Pre-trained Models for End-to-End Image Translation",
    author = "Zhu, Shaolin  and
      Li, Shangjie  and
      Lei, Yikun  and
      Xiong, Deyi",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.751",
    doi = "10.18653/v1/2023.acl-long.751",
    pages = "13433--13447",
    abstract = "Image translation is a task that translates an image containing text in the source language to the target language. One major challenge with image translation is the modality gap between visual text inputs and textual inputs/outputs of machine translation (MT). In this paper, we propose PEIT, an end-to-end image translation framework that bridges the modality gap with pre-trained models. It is composed of four essential components: a visual encoder, a shared encoder-decoder backbone network, a vision-text representation aligner equipped with the shared encoder and a cross-modal regularizer stacked over the shared decoder. Both the aligner and regularizer aim at reducing the modality gap. To train PEIT, we employ a two-stage pre-training strategy with an auxiliary MT task: (1) pre-training the MT model on the MT training data to initialize the shared encoder-decoder backbone network; and (2) pre-training PEIT with the aligner and regularizer on a synthesized dataset with rendered images containing text from the MT training data. In order to facilitate the evaluation of PEIT and promote research on image translation, we create a large-scale image translation corpus ECOIT containing 480K image-translation pairs via crowd-sourcing and manual post-editing from real-world images in the e-commerce domain. Experiments on the curated ECOIT benchmark dataset demonstrate that PEIT substantially outperforms both cascaded image translation systems (OCR+MT) and previous strong end-to-end image translation model, with fewer parameters and faster decoding speed.",
}
```

