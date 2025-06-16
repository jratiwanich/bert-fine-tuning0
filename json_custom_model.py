"""
json_custom_model.py

This script demonstrates how to train your own intent classification model using Hugging Face Transformers,
reading data from a JSON file using pandas.

Steps:
1. Prepare your dataset (JSON with 'text' and 'label' fields)
2. Load and preprocess the data
3. Tokenize the data
4. Set up the model and Trainer
5. Train the model
6. Evaluate the model
7. Save the model
"""

# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch

# 2. Load your dataset (replace 'your_data.json' with your file)
# The JSON should be a list of objects with 'text' and 'label' fields
# Example: [{"text": "How do I reset my password?", "label": "Account access"}, ...]
df = pd.read_json("your_data.json")

# 3. Encode labels as integers
labels = df["label"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# 4. Split into train and test sets
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)

# 5. Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]])

# 6. Load tokenizer and model
model_name = "distilbert-base-uncased"  # You can change this to any supported model
num_labels = len(labels)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
)


# 7. Tokenization function
def tokenize_function(example):
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 8. Set format for PyTorch
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label_id"]
)
test_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label_id"]
)

# 9. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


# 10. Compute metrics
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score

    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# 11. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 12. Train the model
trainer.train()

# 13. Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)

# 14. Save the model
model.save_pretrained("./custom_intent_model_json")
tokenizer.save_pretrained("./custom_intent_model_json")
print("Model and tokenizer saved to ./custom_intent_model_json")
