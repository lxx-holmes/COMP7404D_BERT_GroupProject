# Sentiment Analysis 
Project Name: Question Answering System Based on BERT

### Project Structure
```
├── models
│   └── BERT_SQUAD_QA.ipynb
│   └── BERT_fine_tuning_sentiment_analysis.ipynb
└── README.md
```
- `models`: contains the trained model file.
- `README.md`: the readme file you're currently reading.
———————————
## Model2: BERT Fine-Tuning for Sentiment Analysis

## Description
This project provides a solution for classifying movie comments as positive or negative using the BERT model. The project includes code references, enhancements, and hyperparameters fine-tuning to improve the model's performance.

## Usage Guide
1. Clone the repository to your local machine.
2. Install the required packages listed in the `import necessary libraries` part.
3. Run the codes and check the results. If there is a GPU available in your device, the running speed will be much faster (around half hour).

## Acknowledgment

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
