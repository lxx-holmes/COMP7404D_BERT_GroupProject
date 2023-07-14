# COMP7404D_BERT
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding on sentiment analysis on movie comments and SQuAD datset Q&amp;A based on BERT models 
##
URL: https://github.com/Lingks00/COMP7404D_BERT_GroupProject.git
## Project Structure
```
├── Sentiment Analysis
│   └── BERT_fine_tuning_sentiment_analysis.ipynb
├── SQUAD1.1
│   └── Train_model_GPU_BERT_SQUAD1.1.ipynb
├── SQUAD2.0
│   └── xxxx.ipynb
└── README.md
```
- `models`: contains the trained model file.
- `README.md`: the readme file you're currently reading.
———————————

## Sentiment Analysis 

### Description
This project provides a solution for classifying movie comments as positive or negative using the BERT model. The project includes code references, enhancements, and hyperparameters fine-tuning to improve the model's performance.

### Usage Guide
1. Clone the repository to your local machine.
2. Install the required packages listed in the `import necessary libraries` part.
3. Run the codes and check the results. If there is a GPU available in your device, the running speed will be much faster (around half hour).

### Acknowledgment

We would like to acknowledge the following third-party libraries and websites that were used in this project:

- [Hugging Face Transformers](https://huggingface.co/transformers/): a Python library for natural language processing using pre-trained transformer models such as BERT.
- [PyTorch](https://pytorch.org/): an open-source machine learning library for Python used to build and train neural networks.
- [Pandas](https://pandas.pydata.org/): a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool.
- [scikit-learn](https://scikit-learn.org/stable/): a Python library for machine learning built on top of NumPy, SciPy, and matplotlib.
- [SST2 dataset extracted from IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/): a dataset of 6920 movie reviews from IMDb labeled as positive or negative.

We would also like to thank the authors of the following websites and articles that provided valuable insights and guidance for this project:

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/): a blog post by Jay Alammar that explains the BERT model and its architecture.
- [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/): a tutorial by Chris McCormick that provides a step-by-step guide for fine-tuning the BERT model for a specific task.
- [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583): a research paper by Sunil Kumar Sahu and V. Murahari that proposes a method for fine-tuning the BERT model for text classification tasks.


## SQuAD 1.1
```
 ├── models
 │ └── Train_model_GPU_BERT_SQUAD1.1.ipynb
```
Description:
We fine-tuned the improved Bert model for a open-domain QA system based on SQUAD V1.1. According to evaluation, this model achieve 80% of exact, 87% of f1. Training of BERT models is supported via [Train_model_GPU_BERT_SQUAD1.1.ipynb]

# Reference
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach]
(https://arxiv.org/abs/1907.11692): A research paper about improved Bert model.


## BERT on SQuAD 2.0 Dataset

### Introduction

This project involves **BERT** (Bidirectional Encoder Representations from Transformers), a state-of-the-art natural language processing (NLP) model developed by Google in 2018. It has introduced a pre-trained language representation model that captures the contextual understanding of words and sentences.

**SQuAD 2.0** (Stanford Question Answering Dataset 2.0) is a widely used benchmark dataset for question answering (QA) tasks in natural language processing. SQuAD 2.0 is designed to test the ability of QA systems to not only provide accurate answers but also to identify when no answer is available.

This project implements a Question-Answering model based on BERT, trained and tested with SQuAD 2.0 dataset. The codes are based on BERT QA model provided by Huggingface: https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

### Files

`main.py` main file to run the training and evluation process.

`modeling_bert.py` defines all structures of BERT model.

`utils_squad.py` some utilization methods for SQuAD dataset preprocessing.

`utils_squad_evaluate.py` evaluation methods for the model.

### Environment

You need the following packages installed to run the code:

torch

tqdm

argparse

pytorch_transformers

### Running the code

##### Training

```shell
python main.py --train_file dataset/train.json --predict_file dataset/valid.json --model_type bert --model_name_or_path bert-base-uncased  --output_dir output/ --version_2_with_negative --do_train --do_eval  --do_lower_case --overwrite_output --save_steps 0 --learning_rate 3e-5 --num_train_epochs 2
```

The above command will run the `main.py` to train the model, using `bert-base-uncased` as the pretrained weights, `train.json` as the training set and `valid.json` as validation set. The trainning will be executed with learning rate of **3e-5** for **2** epochs. (Which is the optimal value that we found)

#### Testing

```shell
python main.py --train_file dataset/train.json --predict_file dataset/test.json --model_type bert --model_name_or_path output/  --output_dir output/eval/ --version_2_with_negative --do_eval --do_lower_case
```

The above command will run the `main.py` to evaluate the test set on the model. In the output directory, you can check the output file which contains the answer given by the model to the questions in the test data.

# GLUE 

## Fine Tuning of GLUE in BERT


