# install spacy
#!pip install --upgrade spacy
#!pip install spacy-transformers -q
#!python -m spacy download en
#!-m pip install --upgrade transformers


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


df['word'] = df['word'].map(lambda x: x.lower())
sentences = df.groupby("sentence#")["word"].apply(list).values
labels = df.groupby(by = 'sentence#')['label'].apply(list).values
list(zip(sentences[0], labels[0]))


nlp = spacy.blank('en')
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
epochs = 40
steps = 1
BATCH_SIZE = 12


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
    if float(str(losses['ner'])) < 1.9:
      break

import pickle
with open('/content/spacy_model.pkl', 'wb') as f:
    pickle.dump(nlp, f)