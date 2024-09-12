from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, DataCollatorWithPadding
from datasets import load_from_disk

tokenized_datasets = load_from_disk('./tokenized_dataset')

train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny')

def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
print('trained')

results = trainer.evaluate()
print("Результаты оценки:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
