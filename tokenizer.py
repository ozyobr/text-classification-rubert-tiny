import os
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

dataset_path = './dataset/'

def load_data(folder, label):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                data.append({'text': text, 'label': label})
    return data

neg_data = load_data(os.path.join(dataset_path, 'neg'), 0)
print('neg loaded')
neu_data = load_data(os.path.join(dataset_path, 'neu'), 1)
print('neu loaded')
pos_data = load_data(os.path.join(dataset_path, 'pos'), 2)
print('pos loaded')

all_data = neg_data + neu_data + pos_data
df = pd.DataFrame(all_data)

dataset = Dataset.from_pandas(df)

tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny')

print('tokenizer initializated')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.save_to_disk('./tokenized_dataset')