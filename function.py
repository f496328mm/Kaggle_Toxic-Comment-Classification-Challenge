

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import re
from nltk.stem import SnowballStemmer
from collections import defaultdict  
from collections import Counter
path = '/home/linsam/kaggle/Toxic-Comment-Classification-Challenge'
os.chdir(path)
sys.path.append(path)


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#------------------------------------------------------
from keras.models import Model
#from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

def compare_max_features_and_maxlen(train,mfeature_per,maxlen_per):
    key_word = []
    maxlen = []
    for i in range(len(train)):
        #print(str(i)+'/'+str(len(train)))
        tem = train['comment_text'][i]
        tem_list = re.split(' ',tem)
        [ tem_list.remove('') for i in range(sum( np.array(tem_list) == '' )) ]
        maxlen.append( len(tem_list) )
        key_word.extend(tem_list)
    #--------------------------------------------------------------------
    x = Counter( key_word )
    #dict_x = dict(x)
    #x.most_common()[:5]
    x2 = pd.DataFrame(list(x.items()), columns=['word', 'times'])
    #print(x2['times'].describe())
    x2 = x2.sort_values('times', ascending=False)
    x2.index = range(len(x2))
    
    value = x2['times']
    
    max_features = value#[int(len(value)*mfeature_per)]
    #--------------------------------------------------------------------
    maxlen = int( np.mean(maxlen)*maxlen_per )
    #--------------------------------------------------------------------
    return max_features,maxlen

def compare_target_times(train):
    col_name = ['identity_hate', 'insult', 'obscene','severe_toxic', 'threat', 'toxic']
    for col in col_name:
        y = train[col]
        print(col + ' : ' + str( sum( y==1 ) ))
        
def get_lstm_model(maxlen,max_features,embedding_matrix,embed_size):# output_dim = 1024
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    #from keras.layers import LSTM
    from keras.layers import CuDNNLSTM
    
    model = Sequential()
    model.add(Embedding(max_features, embed_size, weights=[embedding_matrix]))
    #model.add(LSTM(512))
    model.add(CuDNNLSTM(int(embed_size/2)))
    #x = Bidirectional(LSTM(50, return_sequences=True))(x)
    #model.add( GlobalAveragePooling1D() )
    model.add( GlobalMaxPooling1D() )
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def get_cnn_model(maxlen,max_features,embedding_matrix,embed_size):    
    filter_sizes = [1,2,3,5]
    num_filters = 32
    #embed_size = 300
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Reshape((maxlen, embed_size, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    outp = Dense(6, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_gru_model(maxlen,max_features,embedding_matrix,embed_size):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def create_embedding_matrix(embeddings_index,tokenizer,maxlen,max_features,EMBEDDING_FILE):
    embed_size = 300
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
    return embedding_matrix


'''
self = Ensemble(labels,
                 max_features,
                 maxlen,
                 validation_split,
                 batch_size,
                 embed_size,
                 embedding_matrix,
                 vector_train_x,
                 train_y,
                 vector_test_x)
'''
class Ensemble:
    def __init__(self,
                 labels,
                 max_features,
                 maxlen,
                 validation_split,
                 batch_size,
                 embed_size,
                 embedding_matrix,
                 vector_train_x,
                 train_y,
                 vector_test_x,
                 verbose):
        self.labels = labels
        self.max_features = max_features
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.embedding_matrix = embedding_matrix
        self.vector_train_x = vector_train_x
        self.train_y = train_y
        self.vector_test_x = vector_test_x
        self.maxlen = maxlen
        self.epochs = 10
        self.verbose = verbose
    
    def model_fit(self,model,file_path):
        #file_path = "lstm_weights_base.best.hdf5"
        checkpoint = ModelCheckpoint(file_path, 
                                     monitor = 'val_loss', 
                                     verbose = 0, 
                                     save_best_only=True)
        
        early = EarlyStopping(monitor="val_loss")#, patience = 10)
        
        callbacks_list = [checkpoint, early] #early
        #-------------------------------------------
        tem = file_path.replace('_weights_base.best.hdf5','')
        print( tem + ' model fit')
        model.fit(self.vector_train_x,
                  self.train_y,
                  #validation_split = 0.3,
                  validation_split = self.validation_split,
                  batch_size = self.batch_size,
                  epochs = self.epochs,
                  shuffle = False,
                  verbose = self.verbose,
                  callbacks  = callbacks_list)#644us/step
        return model
    
    def build_model(self,get_model,file_path):
        model = get_model(self.maxlen,self.max_features,self.embedding_matrix,self.embed_size)
        #file_path = "lstm_weights_base.best.hdf5"
        model = self.model_fit(model,file_path)
        return model   

    def main(self):
        
        lstm = self.build_model(get_lstm_model,"lstm_weights_base.best.hdf5")
        cnn = self.build_model(get_cnn_model,"cnn_weights_base.best.hdf5")
        gru = self.build_model(get_gru_model,"gru_weights_base.best.hdf5")
        
        models = [lstm,cnn,gru]
        output = ["lstm_baseline.csv","cnn_baseline.csv","gur_baseline.csv"]
        
        pred_set = []
        for i in range(len(models)):
            model = models[i]
            print('predict')
            pred = model.predict(self.vector_test_x,verbose = self.verbose,batch_size = self.batch_size)
            sample_submission = pd.read_csv("data/sample_submission.csv")
            sample_submission[self.labels] = pred
            pred_set.append(sample_submission)
            print('output to ' + output[i] )
            sample_submission.to_csv( 'predict/' + output[i], index=False)        
            # sample_submission.to_csv('predict/123.csv', index=False)   
        #---------------------------------------------------------------------------
        submission = pd.read_csv("sample_submission.csv")
        
        for label in self.labels:
            x = pd.DataFrame( )
            for i in range(len(pred_set)):
                col = 'v' + str(i)
                x[col] = pred_set[i][label] 
            submission[label] = np.mean(x,axis = 1)
            
        submission.to_csv("ensemble.csv", index=False)
       
        
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


def clean_data(data):
    
    replace_text = ['\n','=','!','@',',',r'\"','[ ]{1,10}',r"\)",r"\(",r'\?']
    data['comment_text'] = data['comment_text'].replace(
            to_replace = replace_text,value = ' ',regex=True)
    return data







