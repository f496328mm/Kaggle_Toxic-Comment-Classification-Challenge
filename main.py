
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

import pandas as pd 
import os
import sys
import numpy as np
path = '/home/linsam/kaggle/Toxic-Comment-Classification-Challenge'
os.chdir(path)
sys.path.append(path)
import function
import clean_text

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#------------------------------------------------------
#------------------------------------------------------
print('input data')
EMBEDDING_FILE = '/home/linsam/github/crawl-300d-2M.vec'
train = pd.read_csv("data/train.csv")
# train = train[:10000]
test = pd.read_csv("data/test.csv")
embeddings_index = dict(function.get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

#------------------------------------------------------
print('merge data')
data = train.append(test)
data.index = range(len(data))

print('clean data')
data = function.clean_data(data)

print('get train data')
train = data[ data['identity_hate'].isnull() == 0 ]
#-----------------------------------------------------------
print('compare max features and maxlen')
mfeature_per = 0.00003
maxlen_per = 2
embed_size = 300
value,tem_maxlen = function.compare_max_features_and_maxlen(train,mfeature_per,maxlen_per)
max_features = value[int(len(value)*mfeature_per)]
maxlen = int( np.mean(tem_maxlen)*maxlen_per )
#----------------------------------------------------------
print('data to vector')
vector_train_x,train_y,vector_test_x,tokenizer,labels = clean_text.data2vector(train,test,maxlen,max_features)

print('create embedding matrix')
embedding_matrix = function.create_embedding_matrix(embeddings_index,tokenizer,maxlen,max_features,EMBEDDING_FILE)

#-----------------------------------------------------------------------------
output_dim = 512*2
validation_split = 0.1
batch_size = 512
#-----------------------------------------------------------------------------
print('ensemble predict')
En = function.Ensemble(labels,
                       max_features,
                       maxlen,
                       validation_split,
                       batch_size,
                       embed_size,
                       embedding_matrix,
                       vector_train_x,
                       train_y,
                       vector_test_x,
                       verbose = 0)

En.main()
#-----------------------------------------------------------------------------

# train.




