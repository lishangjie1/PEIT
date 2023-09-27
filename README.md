# PEIT 
## overview
The codes for ACL2023 paper: "PEIT: Bridging the Modality Gap with Pre-trained Models for End-to-End Image Translation" [[paper](https://aclanthology.org/2023.acl-long.751/)]
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

