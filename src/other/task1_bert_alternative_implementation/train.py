# -*- coding: utf-8 -*-
"""bert_ner_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DKp3L_blC26UP-opTnTwjHmPU4Gx6i3F
"""

# !pip install transformers -q

"""## Imports"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import torch
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from transformers import BertTokenizer
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F

data_path = '/content/bert_ner_moountain_dataset.csv'
df = pd.read_csv(data_path, delimiter=";",  error_bad_lines=False )

df.sample(5)

df.columns

df['label'].unique()

df['label'] = df['label'].map(lambda x: x.strip())

df['label'].unique()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_inputs = tokenizer(df["sentence"].tolist(), padding=True, truncation=True, return_tensors="pt")

class CustomDataset(Dataset):
    def __init__(self, tokenized_inputs, labels):
        self.tokenized_inputs = tokenized_inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx])
        }

dataset = CustomDataset(tokenized_inputs, labels=df["label"].map({"Mountain": 1, "O": 0}).tolist())

dataset[0]

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def train(num_epochs=2):
  """
  [num_epochs] 2-4 is recomended for fine-tuning bert
  """
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  optimizer = optim.AdamW(model.parameters(), lr=2e-5)
  criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
      model.train()
      total_loss = 0
      start_time = time.time()
      for batch in dataloader:

          inputs = batch["input_ids"]
          attention_mask = batch["attention_mask"]
          labels = batch['labels']
          outputs = model(inputs, attention_mask=attention_mask)
          loss = criterion(outputs.logits, labels)
          total_loss += loss.item()

          loss.backward()
          optimizer.step()
      end_time = time.time()
      epoch_time = end_time - start_time

      average_loss = total_loss / len(dataloader)
      print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {average_loss}  \nTime: {epoch_time} seconds")

"""
Epoch 1/3 - Average Loss: 0.4773173794016108
Time: 1612.7836396694183 seconds
Epoch 2/3 - Average Loss: 0.6529922951180656
Time: 1605.6941952705383 seconds
Epoch 3/3 - Average Loss: 0.9067454332703943
Time: 1613.9223237037659 seconds
"""
train()

model.eval()

torch.save(model.state_dict(), '/content/model.pth')