# -*- coding: utf-8 -*-
"""vert_ner_spacy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PYujLpStgUEMeddrOfGUTXogaPYJokB9
"""

# install spacy
#!pip install --upgrade spacy
#!pip install spacy-transformers -q
#!python -m spacy download en
#!python -m spacy download en_core_web_lg
#!python -m spacy download en_core_web_trf
#!-m pip install --upgrade transformers
#!python -m spacy download en_trf_bertbaseuncased_lg



import pandas as pd
import numpy as np
from spacy.training.example import Example
import spacy
from spacy.tokens import Doc, Token
from sklearn.model_selection import train_test_split
import random
import time

data_path = '/content/dataset_medium.csv'
df = pd.read_csv(data_path, delimiter=",",  error_bad_lines=False )
print(df.sample(5))
df.info()

sentences = df.groupby("sentence#")["word"].apply(list).values
labels = df.groupby(by = 'sentence#')['label'].apply(list).values

list(zip(sentences[0], labels[0]))

nlp = spacy.blank('en')

# Check if 'ner' pipeline exists, if not, add it
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

TRAIN_DATA = []
for sentence, label in zip(sentences, labels):
    doc = nlp.make_doc(' '.join(sentence))
    ents = [(word.idx, word.idx+len(word), lab) for word, lab in zip(doc, label) if lab != 'O']
    example = Example.from_dict(doc, {"entities": ents})
    TRAIN_DATA.append(example)

optimizer = nlp.initialize()
epochs = 10

steps = 1
BATCH_SIZE = 4
for itn in range(epochs):

    random.shuffle(TRAIN_DATA)
    losses = {}
    step_start = time.time()
    for i, batch in enumerate(spacy.util.minibatch(TRAIN_DATA, size=BATCH_SIZE)):
        nlp.update(batch, sgd=optimizer, losses=losses)
    step_end = time.time()
    step = step_end - step_start
    steps += step
    avg_time_per_step = steps / (i+1)

    print(f"{itn:03d}/{epochs:03d}; step: {round(step, 1)}; loss: {losses}")

save_path = "/content/model"
nlp.to_disk("/content/model")



from google.colab import files
!zip -r /content/spacy_model.zip /content/model
files.download('/content/spacy_model.zip')

def predict_entities(text, model_path):
    # Load the trained model
    nlp = spacy.load(model_path)

    # Process the text
    doc = nlp(text)

    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

text = "I love Everest mountain "
model_path = "/path/to/model"
print(predict_entities(text, model_path))