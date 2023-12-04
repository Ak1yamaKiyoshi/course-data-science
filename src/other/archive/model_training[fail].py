# -*- coding: utf-8 -*-
"""model_training[fail].ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VRC-jgiw_Qr7xDoasOUhRYXVTgva8pDi

# BERT Pre-trained based NER model

### Installing dependencies
- Transformers for BERT pre-trained model
"""

#!pip install pandas transformers tensorflow scikit-learn

"""### Setup"""

import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
from transformers import TFBertModel
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv("/content/stupid_dataset.csv", encoding="latin-1")
df.columns

df["Sentence #"] = df["Sentence #"].fillna(method="ffill")
sentence = df.groupby("Sentence #")[" Word"].apply(list).values
pos = df.groupby(by = 'Sentence #')[' POS'].apply(list).values
tag = df.groupby(by = 'Sentence #')[' Tag'].apply(list).values



df.sample(5)

def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, " POS"] = enc_pos.fit_transform(df[" POS"])
    df.loc[:, " Tag"] = enc_tag.fit_transform(df[" Tag"])

    sentences = df.groupby("Sentence #")[" Word"].apply(list).values
    pos = df.groupby("Sentence #")[" POS"].apply(list).values
    tag = df.groupby("Sentence #")[" Tag"].apply(list).values
    return sentences, pos, tag, enc_pos, enc_tag

sentence,pos,tag,enc_pos,enc_tag = process_data("/content/stupid_dataset.csv")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
MAX_LEN = 128

def tokenize(data, max_len=MAX_LEN):
    input_ids = list()
    attention_mask = list()
    for i in tqdm(range(len(data))):
        encoded = tokenizer.encode_plus(data[i],
                                        add_special_tokens = True,
                                        max_length = MAX_LEN,
                                        is_split_into_words=True,
                                        return_attention_mask=True,
                                        padding = 'max_length',
                                        truncation=True,return_tensors = 'np')


        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
    return np.vstack(input_ids),np.vstack(attention_mask)

X_train,X_test,y_train,y_test = train_test_split(sentence,tag,random_state=42,test_size=0.1)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
X_train.tolist()

input_ids,attention_mask = tokenize(X_train,max_len = MAX_LEN)

val_input_ids,val_attention_mask = tokenize(X_test,max_len = MAX_LEN)

"""# Test padding and truncation lenght"""

# TEST: Checking Padding and Truncation length's
was = list()
for i in range(len(input_ids)):
    was.append(len(input_ids[i]))
set(was)

# Train Padding
train_tag = list()
for i in range(len(y_train)):
    train_tag.append(np.array(y_train[i] + [0] * (128-len(y_train[i]))))

# TEST:  Checking Padding Length
was = list()
for i in range(len(train_tag)):
    was.append(len(train_tag[i]))
set(was)

# Train Padding
test_tag = list()
for i in range(len(y_test)):
    test_tag.append(np.array(y_test[i] + [0] * (128-len(y_test[i]))))

# TEST:  Checking Padding Length
was = list()
for i in range(len(test_tag)):
    was.append(len(test_tag[i]))
set(was)

"""# Building a model"""

# bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def create_model(bert_model,max_len = MAX_LEN):
    input_ids = tf.keras.Input(shape = (max_len,),dtype = 'int32')
    attention_masks = tf.keras.Input(shape = (max_len,),dtype = 'int32')
    bert_output = bert_model(input_ids,attention_mask = attention_masks,return_dict =True)
    embedding = tf.keras.layers.Dropout(0.3)(bert_output["last_hidden_state"])
    output = tf.keras.layers.Dense(17,activation = 'softmax')(embedding)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = [output])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

#with strategy.scope():
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
model = create_model(bert_model,MAX_LEN)

model.summary()

tf.keras.utils.plot_model(model)

"""# Training model"""

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(mode='min',patience=5)
history_bert = model.fit(
    [input_ids,attention_mask],
    np.array(train_tag),
    validation_data = ([val_input_ids,val_attention_mask],np.array(test_tag)),
    epochs = 2,
    batch_size = 30*2,
    callbacks = early_stopping,
    verbose = True
)

model.save_weights("ner_bert_weights")

import matplotlib.pyplot as plt
plt.plot(history_bert.history['accuracy'])
plt.plot(history_bert.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history_bert.history['loss'])
plt.plot(history_bert.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

def pred(val_input_ids,val_attention_mask):
    return model.predict([val_input_ids, val_attention_mask])



def testing(val_input_ids, val_attention_mask, enc_tag, y_test):
    val_input = val_input_ids.reshape(1,128)
    val_attention = val_attention_mask.reshape(1,128)

    # Print Original Sentence
    sentence = tokenizer.decode(val_input_ids[val_input_ids > 0])
    print("Original Text : ",str(sentence))
    print("\n")
    true_enc_tag = enc_tag.inverse_transform(y_test)

    print("Original Tags : " ,str(true_enc_tag))
    print("\n")

    prediction = pred(val_input,val_attention)
    pred_with_pad = np.argmax(prediction,axis = -1)
    print(prediction)
    print(pred_with_pad)
    pred_without_pad = pred_with_pad[pred_with_pad>0]
    pred_enc_tag = enc_tag.inverse_transform(pred_without_pad)

    print("Predicted Tags : ",pred_enc_tag)

# Example usage
index = 0
testing(val_input_ids[index], val_attention_mask[index], enc_tag, y_test[index])
