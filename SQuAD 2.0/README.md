# BERT on SQuAD 2.0 Dataset

## Introduction

This project involves **BERT** (Bidirectional Encoder Representations from Transformers), a state-of-the-art natural language processing (NLP) model developed by Google in 2018. It has introduced a pre-trained language representation model that captures the contextual understanding of words and sentences.

**SQuAD 2.0** (Stanford Question Answering Dataset 2.0) is a widely used benchmark dataset for question answering (QA) tasks in natural language processing. SQuAD 2.0 is designed to test the ability of QA systems to not only provide accurate answers but also to identify when no answer is available.

This project implements a Question-Answering model based on BERT, trained and tested with SQuAD 2.0 dataset. The codes are based on BERT QA model provided by Huggingface: https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

## Files

`main.py` main file to run the training and evluation process.

`modeling_bert.py` defines all structures of BERT model.

`utils_squad.py` some utilization methods for SQuAD dataset preprocessing.

`utils_squad_evaluate.py` evaluation methods for the model.

## Environment

You need the following packages installed to run the code:

torch

tqdm

argparse

pytorch_transformers

## Running the code

#### Training

```shell
python main.py --train_file dataset/train.json --predict_file dataset/valid.json --model_type bert --model_name_or_path bert-base-uncased  --output_dir output/ --version_2_with_negative --do_train --do_eval  --do_lower_case --overwrite_output --save_steps 0 --learning_rate 3e-5 --num_train_epochs 2
```

The above command will run the `main.py` to train the model, using `bert-base-uncased` as the pretrained weights, `train.json` as the training set and `valid.json` as validation set. The trainning will be executed with learning rate of **3e-5** for **2** epochs. (Which is the optimal value that we found)

#### Testing

```shell
python main.py --train_file dataset/train.json --predict_file dataset/test.json --model_type bert --model_name_or_path output/  --output_dir output/eval/ --version_2_with_negative --do_eval --do_lower_case
```

The above command will run the `main.py` to evaluate the test set on the model. In the output directory, you can check the output file which contains the answer given by the model to the questions in the test data.