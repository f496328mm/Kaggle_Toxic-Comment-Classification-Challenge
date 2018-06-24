
# https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-069

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
path = '/home/linsam/kaggle/Toxic-Comment-Classification-Challenge'
os.chdir(path)
sys.path.append(path)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output().decode("utf8"))

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
maxlen = 100


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#train = train.sample(frac=1)

train_x = train["comment_text"].fillna("xxxxxxxx").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_y = train[list_classes].values
test_x = test["comment_text"].fillna("xxxxxxxx").values


tokenizer = text.Tokenizer(num_words=max_features)
#num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
tokenizer.fit_on_texts(list(train_x))

list_tokenized_train = tokenizer.texts_to_sequences(train_x)
list_tokenized_test = tokenizer.texts_to_sequences(test_x)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
# maxlen 序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_model():
    #embed_size = 100
    inp = Input(shape=(66, ))
    x = Embedding(max_features, 66)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    #x = Dropout(0.1)(x)
    #x = Dense(50, activation="relu")(x)
    #x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()
# model.summary()
batch_size = 512*8
epochs = 30


file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, 
                             save_best_only=True)

early = EarlyStopping(monitor="val_loss", patience = 20)


callbacks_list = [checkpoint, early] #early

model.fit(X_t, train_y, 
          batch_size = batch_size, 
          epochs = epochs, 
          validation_split = 0.3, 
          callbacks  = callbacks_list)

#model.load_weights(file_path)

y_test = model.predict(X_te,verbose=1,batch_size = batch_size)



sample_submission = pd.read_csv("sample_submission.csv")

sample_submission[list_classes] = y_test



sample_submission.to_csv("baseline.csv", index=False)





