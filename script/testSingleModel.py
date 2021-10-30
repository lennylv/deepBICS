import numpy as np
import os,sys
import fnmatch
import xlwt
from SingleModel import get_model
import tensorflow as tf
from scipy.stats import pearsonr
from scipy.stats import spearmanr

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

rootDir = "/home/lzhpc/user/xiaoyu/binding_improve/dream2017withmse/"
testDir = rootDir + "regression/test/"
preDir = rootDir + "regression/single_pre/"
modelDir = rootDir + "model/CTCF_H1-hESC_trainBest.h5"

serial_model=get_model()

e = xlwt.Workbook()
sheet1 = e.add_sheet(u'sheet1',cell_overwrite_ok=True)
i = 1

# for testF in ["CEBPB_IMR-90.npy"]:
for testF in fnmatch.filter(os.listdir(testDir),"*.npy"):
    front_name = testF.split(".")[0]
    # model = modelDir + front_name + '_valiBest.h5'
    testData = np.load(testDir+testF)
    seq = testData[:,2:195]
    DNase = testData[:,195:388]
    y = testData[:,-1]
    if os.path.exists(modelDir):    
        try:
            serial_model.load_weights(modelDir)
            y_pre = serial_model.predict([seq,DNase])
            # np.save(preDir + testF,y_pre)
            y = y.astype(np.float32)
            y = np.reshape(y,(len(y),1))
            y = np.squeeze(y)
            y_pre = np.squeeze(y_pre)
            # import pdb; pdb.set_trace()
            pear = pearsonr(y,y_pre)
            spearman = spearmanr(y,y_pre)
            print(pear[0],spearman[0])
            sheet1.write(i,0,front_name)
            sheet1.write(i,1,str(pear[0]))
            sheet1.write(i,2,str(spearman[0]))
        except ValueError:
            pass
    i += 1
e.save("single_test_cnnrnn.xls")
