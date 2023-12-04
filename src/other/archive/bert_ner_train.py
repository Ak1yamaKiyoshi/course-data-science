# -*- coding: utf-8 -*-
"""bert_ner_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n0J0wynDUSJnT5gvJgsVSIpG3k40rWw0
"""

!pip install transformers -q

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

data_path = '/content/dataset.csv'
df = pd.read_csv(data_path, delimiter=",",  error_bad_lines=False )

df.sample(5)

df.columns

df['label'].unique()

sentence = df.groupby("sentence#")["word"].apply(list).values
label = df.groupby(by = 'sentence#')['label'].apply(list).values

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_sentence = []
MAX_LEN = 128
for i in sentence:
  tokenized_inputs = tokenizer.encode_plus(i,
                                        add_special_tokens = True,
                                        max_length = MAX_LEN,
                                        is_split_into_words=True,
                                        return_attention_mask=True,
                                        padding = 'max_length',
                                        truncation=True,return_tensors = 'np')
  tokenizer(i, padding=True, truncation=True, return_tensors="pt")
  encoded_sentence.append(tokenized_inputs)

#sentence, encoded_sentence

class CustomDataset(Dataset):
    def __init__(self, sentence, labels, encoded_sentences, max_len):
        self.tokenized_inputs = encoded_sentences
        self.labels           = labels
        self.sentences        = sentence
        self.map = {"O": 0, "Mountain": 1}
        self.reversal_map = {0: "O",  1: "Mountain"}
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lbls = list(map(lambda x: self.map[x], self.labels[idx]))
        return {
          "inputs": self.tokenized_inputs[idx],
          "labels": np.array(lbls + [0] * (self.max_len - len(lbls))),
          "sentence": np.array(self.sentences[idx] + [0] * (self.max_len - len(self.sentences[idx]))),
        }

dataset = CustomDataset(sentence, label, encoded_sentence, MAX_LEN)

print(dataset[0]["labels"])
print(dataset[0]["sentence"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def train(num_epochs=2):
  """
  [num_epochs] 2-4 is recomended for fine-tuning bert
  """

  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
  print("000")
  optimizer = optim.AdamW(model.parameters(), lr=2e-5)
  criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
      model.train()
      total_loss = 0
      start_time = time.time()
      for batch in dataloader:
          inputs = np.array(batch["inputs"]["input_ids"])
          attention_mask = np.array(batch["inputs"]["attention_mask"])
          labels = np.array(batch['labels'])
          print(inputs)

          outputs = model(inputs, attention_mask=attention_mask)
          loss = criterion(outputs.logits, labels)
          total_loss += loss.item()

          loss.backward()
          optimizer.step()
      end_time = time.time()
      epoch_time = end_time - start_time

      average_loss = total_loss / len(dataloader)
      print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {average_loss}  \nTime: {epoch_time} seconds")

train()

model.eval()

torch.save(model.state_dict(), '/content/model.pth')

def predict_sentence(sentence_x, model, tokenizer):
    """
      [sentence_x] - String sentence to predict
      [model] - BERT model that will predict on sentence
      [tokenizer] - tokenizer used for that model
    """
    tokenized_inputs = tokenizer(sentence_x.split(" "), padding=True, truncation=True, return_tensors ="pt")
    with torch.no_grad():
      outputs = model(**tokenized_inputs)
      predictions = torch.argmax(outputs.logits, dim=1)

      predicted_labels = ["Mountain" if pred > 0.7 else "." for pred in predictions.tolist()]
      return predicted_labels


sentences_to_predict = [
    "expedition to Ukraine is cool, i love Svydovec, Alps"

    ]


for sentence_x in sentences_to_predict:
  predicted_labels =  predict_sentence(sentence_x, model, tokenizer)
  for i in zip(sentence_x.split(" "), predicted_labels):
    print(f"{i[0]}({i[1]}) ", end="")
  print("")

i = 0
while True:
  print(f"slept well {i}")
  time.sleep(5)
  i+= 5