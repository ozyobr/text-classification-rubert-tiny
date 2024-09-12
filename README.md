# Text Classification with rubert-tiny
This repository contains a project for fine-tuning the [RuBERT-Tiny](https://huggingface.co/cointegrated/rubert-tiny) model for text classification. The model is trained on a custom dataset specifically designed for multi-class text classification, with a focus on determining the sentiment of movie reviews. By using this model, we aim to build an efficient system for automatic sentiment analysis of user reviews, which can help in summarizing public opinion about movies.

## Dataset
The dataset used for fine-tuning the model consists of user comments classified into three categories: positive, neutral, and negative.

### Dataset Source
The dataset was taken from https://www.kaggle.com/datasets/mikhailklemin/kinopoisks-movies-reviews/data

Alternatively, you can use your own dataset following the format described below.

### Dataset Format
The dataset should be in the following format (CSV or JSON), with each row representing a text sample and its label:
- `text`: The text content (string)
- `label`: The corresponding label (integer, 0 for negative, 1 for neutral, 2 for positive)

## Installation and startup
### Install Required Libraries

```pip install -r requirements.txt```

### Tokenize dataset
If you are using the dataset from the example, you can download the tokenized version of the dataset [from this link](https://drive.google.com/file/d/1JQshNd7z-xW4zFkBpDbKCUwaORUCpoOx/view?usp=sharing). This will save you time by skipping the tokenization step.

If you are using a custom dataset, you need to tokenize it yourself. To do this, run the `tokenize.py` script provided in the repository. This script will load your dataset, tokenize it, and save the tokenized dataset to disk.

### Fine-tuning model
Run `fine-tuning.py` to fine-tune the model

##
## Evaluation Results

The fine-tuned RuBERT-Tiny model was evaluated on the test dataset, and the following metrics were obtained:

| Metric         | Value  |
|----------------|--------|
| **Loss**       | 0.6213 |
| **Accuracy**   | 74.00% |
| **Precision**  | 71.25% |
| **Recall**     | 74.00% |
| **F1-Score**   | 71.70% |

These results indicate that the model performs well for the task of sentiment classification, balancing precision and recall to achieve a solid F1-score.


##
You can download a ready model, fine-tuned on the dataset from the example at the [link](https://drive.google.com/file/d/1Jq83rtq8vHSARxO5LRXlXVqphveGuZ9G/view?usp=sharing)
