from __future__ import print_function
import os
import sys
import numpy as np
import sys
import pickle
from pearson import pearson
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Input
from tensorflow.keras.layers import Average,Embedding,BatchNormalization,Conv1D,Dropout,Flatten,Concatenate,Dense,Attention,Reshape,Bidirectional,GRU,LSTM,TimeDistributed
from tensorflow.keras.utils import plot_model
from highway_fcn import Highway_fcn
import AttLayer
# get number of TF and CL
ListTFpath = '/home/lzhpc/user/xiaoyu/binding_improve/dream2017withAttn/TFlist.pkl'
ListCLpath = '/home/lzhpc/user/xiaoyu/binding_improve/dream2017withAttn/CLlist.pkl'
listTF = pickle.load(open(ListTFpath,'rb'))
listCL = pickle.load(open(ListCLpath,'rb'))
tfNum = len(listTF)
clNum = len(listCL)
del listTF,ListTFpath
del listCL,ListCLpath

seqLen = 193
wordMax = 65536 # 8-mer word
# building the learning model
wordVecDim = 64
cnn1filters = 128
cnn1win = 16
cnn2filters = 128
cnn2win = 16
poolsize = 2
rnn1Dim = 128
dense3Dim = 16
dense1Dim = 128
dense2Dim = 128#1024#16

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

def gelu(input_tensor):
	cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.math.sqrt(2.0)))
	return input_tensor*cdf

def model(inputSeq,inputDNase):
    inputDNase1 = Reshape((seqLen,1))(inputDNase)
    
    # old_model = keras.models.load_model("/home/lzhpc/user/xiaoyu/NewTerm/log16_binary/log16valiBestModel.h5")
    # embed = Embedding(wordMax,wordVecDim,weights=old_model.layers[1].get_weights(),trainable=False)(inputSeq)
    embed = Embedding(wordMax,wordVecDim)(inputSeq)
    embed = BatchNormalization()(embed)
    inputs = Concatenate()([embed,inputDNase1])
    toCnn = Dense(dense3Dim, activation='relu')(inputs)
    cnn1 = Conv1D(cnn1filters,kernel_size=cnn1win,padding='valid' ,activation='relu')(toCnn)
    cnn1 = Dropout(0.3)(cnn1)
    cnn2 = Conv1D(cnn2filters,kernel_size=cnn2win,padding='valid',activation='relu')(cnn1)
    cnn2 = Dropout(0.3)(cnn2)
    # outcnn = AttLayer.AttLayer(128)(cnn1)
    rnn1 = Bidirectional(layers.GRU(rnn1Dim,return_sequences=True),merge_mode='sum')(cnn2)
    rnn1 = Dropout(0.3)(rnn1)
    rnn2 = Bidirectional(layers.GRU(rnn1Dim,return_sequences=True),merge_mode='sum')(rnn1)
    rnn2 = Dropout(0.3)(rnn2)
    attn = AttLayer.AttLayer(128)(rnn2)
    # h_rnn1 = Highway_fcn(128)(rnn1)
    # h_rnn1 = layers.MaxPool1D(pool_size=poolsize, strides=poolsize, padding='valid')(rnn2)
    # seq = Flatten()(h_rnn1) 

    # toDense = Concatenate()([attn,outcnn])
    dense1 = Dense(dense1Dim, activation='relu')(attn)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(dense2Dim, activation='relu')(dense1)
    dense2 = Dropout(0.5)(dense2)
    y = Dense(1, activation='sigmoid')(dense2)
    # y = gelu(y)
    return y

def get_model():
    inputSeq = Input(shape=(seqLen,))
    inputDNase = Input(shape=(seqLen,))
    
    y = model(inputSeq,inputDNase)

    serial_model = keras.Model(inputs=[inputSeq,inputDNase], outputs=y)
    serial_model.summary()
    adam=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9,beta_2=0.999,epsilon=1e-08)
    serial_model.compile(loss='mse',optimizer=adam,metrics=[pearson])
    return serial_model