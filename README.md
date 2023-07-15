# COMP7404D_BERT
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding on sentiment analysis on movie comments and SQuAD datset Q&amp;A based on BERT models 
## Group Members:

 Jiang Qiyue
 
 Jiang Wenxuan
 
 Lau Ling Kei
 
 Li Xiaoxu
 
 Zong Chen
 
## Experiments
In this projects, we fine-tuned BERT model for 4 tasks:
1. SST2 dataset for sentiment analysis on movie review text
2. SQUAD1.1 dataset for question-answering
3. SQUAD2.0 dataset for question-answering
4. GLUE-CoLA dataset for sequence classification task
##
URL: https://github.com/Lingks00/COMP7404D_BERT_GroupProject.git
## Project Structure
```
├── Sentiment Analysis
│   └── BERT_fine_tuning_sentiment_analysis.ipynb
│   └── BERT_fine_tuning_sentiment_analysis_submit.pdf
│   └── testing_data.csv
├── SQUAD1.1
│   └── Train_model_GPU_BERT_SQUAD1.1.ipynb
│   └── compute_answers.py
│   └── data_loading
│         └── qa_dataset.py   _init_.py   utils.py
│   └──evaluation
│         └── evaluate.py   _init_.py   utils.py
│   └──models
│         └── span_models.py   _init_.py   utils.py
│   └──squad_data
│         └── data_model.py   _init_.py   utils.py   parser.py
│         └── dev-v1.1.json   predictions.txt   training_set.json
│   └──utils
│         └── embedding_utils.py   _init_.py   utils.py


├── SQUAD2.0
│   └── main.py
│   └── modeling_bert.py
│   └── dataset
│         └── test.json   train.json   valid.json
│   └── utils
│         └── utils_squad.py   utils_squad_evaluate.py
│   └── output
│         └── predictions_.json
├── GLUE
│   └── BERT_GLUE.ipynb
│   └── GLUE VID.zip
└── README.md
```

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
- Bansal, L. (2021, Sep 18). Fine-Tuning BERT for Text Classification in PyTorch [Blog post]. Retrieved from https://luv-bansal.medium.com/fine-tuning-bert-for-text-classification-in-pytorch-503d97342db2 Alammar, J. (2019, Nov 26). 
- A Visual Guide to Using BERT for the First Time [Web article]. Retrieved from http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/


## BERT on SQuAD 1.1 Dataset
### Description
We fine-tuned the improved Bert model for a open-domain QA system based on SQUAD V1.1. According to evaluation, this model achieve 80% of exact, 87% of f1. 
Stanford Question Answering Dataset is a collection of 100k crowdsourced question/answer pairs. In order to compare the model performance, first we use squad v1.1 as training dataset.

### Usage Guide
1. Clone the repository to your local machine.
2. Install the required packages .
3. Run the codes in Train_model_GPU_BERT_SQUAD1.1.ipynb. 

#### Files
Train_model_GPU_BERT_SQUAD1.1.ipynb: main jupyter notebook to fine-tune BERT model
Utils.py: utilization methods
_Init_.py: initialization methods
evaluate.py: Official evaluation script for SQuAD
qa_dataset.py: custom text dataset for BERT model
span_models.py: get answer span
data_model.py: define data unit structure
parser.py: Parser for SQuAD format datasets.
embedding_utils.py: download pre-trained model

###Environment
To setup the environment, please run the following commands:
conda create -n squad_qa_pytorch python=3.8
conda activate squad_qa_pytorch
conda install pytorch cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=squad_qa_pytorch

### Acknowledgment
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach]
(https://arxiv.org/abs/1907.11692): A research paper about improved Bert model.







## BERT on SQuAD 2.0 Dataset

### Description

This project involves **BERT** (Bidirectional Encoder Representations from Transformers), a state-of-the-art natural language processing (NLP) model developed by Google in 2018. It has introduced a pre-trained language representation model that captures the contextual understanding of words and sentences.

**SQuAD 2.0** (Stanford Question Answering Dataset 2.0) is a widely used benchmark dataset for question answering (QA) tasks in natural language processing. SQuAD 2.0 is designed to test the ability of QA systems to not only provide accurate answers but also to identify when no answer is available.

This project implements a Question-Answering model based on BERT, trained and tested with SQuAD 2.0 dataset. 

### Usage Guide
#### Files

`main.py` main file to run the training and evluation process.

`modeling_bert.py` defines all structures of BERT model.

`utils_squad.py` some utilization methods for SQuAD dataset preprocessing.

`utils_squad_evaluate.py` evaluation methods for the model.

#### Environment

You need the following packages installed to run the code:

torch

tqdm

argparse

pytorch_transformers

#### Running the code

##### Training

```shell
python main.py --train_file dataset/train.json --predict_file dataset/valid.json --model_type bert --model_name_or_path bert-base-uncased  --output_dir output/ --version_2_with_negative --do_train --do_eval  --do_lower_case --overwrite_output --save_steps 0 --learning_rate 3e-5 --num_train_epochs 2
```

The above command will run the `main.py` to train the model, using `bert-base-uncased` as the pretrained weights, `train.json` as the training set and `valid.json` as validation set. The trainning will be executed with learning rate of **3e-5** for **2** epochs. (Which is the optimal value that we found)

##### Testing

```shell
python main.py --train_file dataset/train.json --predict_file dataset/test.json --model_type bert --model_name_or_path output/  --output_dir output/eval/ --version_2_with_negative --do_eval --do_lower_case
```

The above command will run the `main.py` to evaluate the test set on the model. In the output directory, you can check the output file which contains the answer given by the model to the questions in the test data.

### Acknowledgment
The codes are based on BERT QA model provided by Huggingface: https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

## GLUE 

### Description
* Use BertForSequenceClassification pre-trained and pre-modificated model
* Use GLUE-CoLA dataset
For this task, we first want to modify the pre-trained BERT model to give outputs for classification, and then we want to continue training the model on our dataset until that the entire model, end-to-end, is well-suited for our task.

We'll be using BertForSequenceClassification. This is the normal BERT model with an added single linear layer on top for classification that we will use as a sentence classifier. As we feed input data, the entire pre-trained BERT model and the additional untrained classification layer is trained on our specific task.

### Usage Guide
Run the BERT_GLUE.ipynb file on Colab
### Acknowledgment
The pre-trained model is BertForSequenceClassification from Huggingface:
https://huggingface.co/docs/transformers/model_doc/bert#bertforsequenceclassification
