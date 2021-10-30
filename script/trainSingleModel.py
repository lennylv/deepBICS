from __future__ import print_function
import os
import sys
import numpy as np
import sys
import pickle
import h5py
import fnmatch
import random
from SingleModel import get_model
from transformer import get_model1
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

maxEpoch = 100
logPath = "/home/lzhpc/user/xiaoyu/binding_improve/dream2017withmse/tensor_logs/"
trainDir = "/home/lzhpc/user/xiaoyu/binding_improve/dream2017withmse/regression/train/"
OutputDir = '/home/lzhpc/user/xiaoyu/binding_improve/dream2017withmse/model/'

trds = sys.argv[1]

name = trds.split(".")[0]
trainFn = trainDir+trds
trainBestModel = OutputDir + name +'_trainBest.h5'
valiBestModel = OutputDir + name +'_valiBest.h5'

# logically load all into memory
trainDS = np.load(trainFn)
np.random.shuffle(trainDS)
dsLen = trainDS.shape[0]
validSplit = 0.1
batchSize = 64
trainLen = int(dsLen * (1-validSplit))
trainBatchesNum = int(trainLen/batchSize)

serial_model=get_model()

checkpoiner = keras.callbacks.ModelCheckpoint(filepath=valiBestModel,verbose = 1,save_best_only=True,save_weights_only=True)
reduceLRpatience = 2
earlyStopPatience = reduceLRpatience*10
reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=reduceLRpatience)
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=earlyStopPatience,mode='auto')
tensorboard = keras.callbacks.TensorBoard(os.path.join(logPath,name),write_graph=True)

serial_model.fit([trainDS[:,2:195],trainDS[:,195:388]],
                  trainDS[:,-1],
                  batch_size=batchSize,
                  epochs = maxEpoch,
                  callbacks = [checkpoiner,reduceLR,earlystopping],
                  shuffle=True,
                  validation_split=validSplit
                  )

serial_model.save_weights(trainBestModel,save_format='h5')