from __future__ import print_function
import os
import sys
import numpy as np
import sys
import pickle
from pearson import pearson
from tensorflow import keras
from tensorflow.keras import layers,Input
from tensorflow.keras.layers import Average,Embedding,BatchNormalization,Conv1D,Dropout,Flatten,Concatenate,Dense,Attention,Reshape,Bidirectional,GRU,LSTM,TimeDistributed
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from Encoder import Encoder
from AttLayer import AttLayer
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
wordVecDim = 8
cnn1filters = 128#64
cnn1win = 16
cnn2filters = 64#16
cnn2win = 8
poolsize = 8#4
dense1Dim = 64
dense2Dim = 64

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

def model(inputSeq,inputDNase):
    
    # old_model = keras.models.load_model("/home/lzhpc/user/xiaoyu/NewTerm/log16_binary/log16valiBestModel.h5")
    # embed = Embedding(wordMax,wordVecDim,weights=old_model.layers[1].get_weights(),trainable=False)(inputSeq)
    embed = Embedding(wordMax,wordVecDim)(inputSeq)
    embed = BatchNormalization()(embed)
    # inputs = Concatenate()([embed,inputDNase1])
    
    cnn1 = Conv1D(cnn1filters,kernel_size=cnn1win,padding='valid' ,activation='relu')(embed)
    cnn1 = Dropout(0.2)(cnn1)
    cnn2 = Conv1D(cnn2filters,kernel_size=cnn2win,padding='valid',activation='relu')(cnn1)
    cnn2 = Dropout(0.2)(cnn2)
    mixFeature = Encoder(2, 64, 4, 256, rate=0.5)(cnn2)
    attn = AttLayer(64)(mixFeature)

    dense1 = Dense(dense1Dim, activation='relu')(attn)
    dense1 = Dropout(0.3)(dense1)
    dense2 = Dense(dense2Dim, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    y = Dense(1,activation='sigmoid')(dense2)
    return y

def get_model1():
    inputSeq = Input(shape=(seqLen,))
    inputDNase = Input(shape=(seqLen,))
    
    y = model(inputSeq,inputDNase)

    serial_model = keras.Model(inputs=[inputSeq,inputDNase], outputs=y)
    serial_model.summary()
    adam=keras.optimizers.Adam(learning_rate=1e-04, beta_1=0.9,beta_2=0.999,epsilon=1e-08)
    serial_model.compile(loss='mse',optimizer=adam,metrics=['mae',pearson])
    return serial_model