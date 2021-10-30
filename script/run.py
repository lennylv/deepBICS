import os,fnmatch

trainDir = "/home/lzhpc/user/xiaoyu/binding_improve/dream2017withmse/regression/train/"

# for trds in fnmatch.filter(os.listdir(trainDir),"*.npy"):
for trds in ["CTCF_H1-hESC.npy","CTCF_MCF-7.npy","CTCF_K562.npy","CTCF_IMR-90.npy","REST_HeLa-S3.npy","REST_MCF-7.npy","CREB1_HepG2.npy"]:
    os.system("python trainSingleModel.py " + trds)
