# MCL-VD: Multi-modal Contrastive Learning with LoRA-Enhanced GraphCodeBERT for Effective Vulnerability Detection

## Overview
This repository contains the code for the paper **"MCL-VD: Multi-modal Contrastive Learning with LoRA-Enhanced GraphCodeBERT for Effective Vulnerability Detection"**. 

## Prerequisites
To run this project, you need to install the following dependencies:
1. **tree-sitter**: A library to parse the source code and generate ASTs.
2. **transformers**: The HuggingFace `transformers` library to use pre-trained models such as `GraphCodeBERT`.
3. **openai**: Required for generating code comments using GPT-4 (or GPT-4 Mini).
You can install them using pip:
```bash
pip install tree-sitter transformers openai
```

### Datasets
You need to download the following datasets for training and testing:Devign Reveal Big-Vul. https://drive.google.com/drive/u/0/folders/1NHPepk-6zqUTrV_B7wdFfdaRlzP5iXJg
Once the datasets are downloaded, extract them to the project directory.


### Data Preprocessing
After downloading and extracting the datasets, you need to process the data:

step 1: Navigate to the data_processing folder and run the dataprocess.py file to generate AST trees and source code with comments.
```
python data_processing/dataprocess.py
```
step 2: After generating the AST and code, move to the gen_comments folder and run gpt4comment.py to generate the comments for the source code.
```
python gen_comments/gpt4comment.py
```

step 3: Finally, run the filter.py file to remove comments from the source code, preparing the code for model training.
```
python data_processing/filter.py
```
### Running the Models
Once the data is preprocessed, you can train and test the models for different datasets.
To train and test on Devign, Reveal, and Big-Vul, simply run the following scripts:
Please note that the model here requires downloading graphcodebert-base, please refer to the link: https://huggingface.co/transformers/v4.2.2/index.html。
```angular2html
python model/RQ1/devign_main.py
```
```angular2html
python model/RQ1/reveal_main.py
```
```angular2html
python model/RQ1/bigvul_main.py
```
RQ2, RQ3, RQ4: For experiments in RQ2, RQ3, and RQ4, run the corresponding Python scripts for the respective dataset:RQ2bigvul.py, RQ2devign.py, RQ2reveal.py RQ3bigvul.py, RQ3Devign.py, RQ3reveal.py RQ4bigvul.py, RQ4devign.py, RQ4reveal.py 
Example:
```angular2html
python RQ2/RQ2devign.py
```
RQ5: To evaluate the impact of supplementing comments during preprocessing, you can compare the performance of the model using the datasets both with and without the generated comments. Simply run the models on the preprocessed datasets with and without comments and compare the results.

