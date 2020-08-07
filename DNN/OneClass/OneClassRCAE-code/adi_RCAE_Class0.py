## EBOV Abs Example
### RCAE

import os.path
from src.models.RCAEnw import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "adi"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128

Cfg.NUMoutliers = 15220
Cfg.epochIn = 2  #  No. epoch
#Cfg.epochIn = 100  #  No. epoch
#Cfg.FitBatchSize = 50
Cfg.FitBatchSize = 500
Cfg.fracVal = .9628
Cfg.adi_normal = 0  # 0 (default), means that Ab fingerprints in SITE1 sets (class 0) are the normal class 
#Cfg.adi_normal = 1 #  Ab fingerprints (from various Abs) in SITE2 sets are treated as normal

# /home_USER must point to user's home directory
PROJECT_DIR = "/home_USER/AbsFngP/DNN/OneClass/OneClassRCAE-code/"

USE_DATASET = "L1"
Cfg.ADI_DATA_IN = PROJECT_DIR + "/../ADI_DB/data" + USE_DATASET +"/"
REPORT_DIR =       PROJECT_DIR + "/reports/figures/" + "Clss" + str(Cfg.adi_normal) + "ADI/"
REPORT_RCAE_DIR =  REPORT_DIR + USE_DATASET + "_RCAE/"
REPORT_DIR_BEST =  REPORT_RCAE_DIR + "best/"
REPORT_DIR_WORST = REPORT_RCAE_DIR + "worst/"
MODEL_SAVE_PATH =  REPORT_RCAE_DIR 

if not os.path.isdir(REPORT_DIR):
   os.makedirs(REPORT_DIR) 

if not os.path.isdir(REPORT_RCAE_DIR):
   os.makedirs(REPORT_RCAE_DIR) 
   os.makedirs(REPORT_DIR_BEST) 
   os.makedirs(REPORT_DIR_WORST) 

REPORT_SAVE_PATH = REPORT_RCAE_DIR

Cfg.MYDATA = "ADI"
Cfg.MYMETHOD = "RCAE"
Cfg.REPORT_OUTDIR = REPORT_RCAE_DIR
Cfg.SAVE_MODEL_DIR = MODEL_SAVE_PATH

PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
RANDOM_SEED = [89]
AUC = []
for seed in RANDOM_SEED:  
  Cfg.seed = seed
  rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  print("Train Data Shape: ",rcae.data._X_train.shape)
  print("Train Label Shape: ",rcae.data._y_train.shape)
  print("Validation Data Shape: ",rcae.data._X_val.shape)
  print("Validation Label Shape: ",rcae.data._y_val.shape)
  print("Test Data Shape: ",rcae.data._X_test.shape)
  print("Test Label Shape: ",rcae.data._y_test.shape)
  print("===========TRAINING AND PREDICTING WITH DCAE============================")
  auc_roc = rcae.fit_and_predict()
  print("========================================================================")
  AUC.append(auc_roc)
  
print("===========END TRAINING AND PREDICTING WITH DCAE==========================")
print("AUROC computed ", AUC)
auc_roc_mean = np.mean(np.asarray(AUC))
auc_roc_std = np.std(np.asarray(AUC))
print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
print("========================================================================")
