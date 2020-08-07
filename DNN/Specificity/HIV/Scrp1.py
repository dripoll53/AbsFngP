# PY SCRIPT FOR HIV DNNs
import os.path
from os import path
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.callbacks
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import re
import pandas as pd
import ntpath
from sklearn.metrics import confusion_matrix

HEIGHT = 150
WIDTH = 150

base_model = ResNet50(weights='imagenet', 
                       include_top=False, 
                       input_shape=(HEIGHT, WIDTH, 3))


BATCH_SIZE = 100

TRAIN_DIR = "/home_USER/AbsFngP/DNN/Specificity/HIV/data1/train/"

train_datagen =  ImageDataGenerator(
       preprocessing_function=preprocess_input,
       rotation_range=65,
       horizontal_flip=False,
       vertical_flip=False
     )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=BATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode="categorical",
                                                  #class_mode="binary",
                                                  shuffle=True, seed=68)


# print (len(train_generator))
num_train_images = len(train_generator) * BATCH_SIZE  * 30
print ('num_train_images=' , num_train_images)

VALID_DIR = "/home_USER/AbsFngP/DNN/Specificity/HIV/data1/valid/"

valid_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      horizontal_flip=False,
      vertical_flip=False
     )

VBATCH_SIZE = BATCH_SIZE 
valid_generator = valid_datagen.flow_from_directory(VALID_DIR, 
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=VBATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode="categorical",
                                                  shuffle=True, seed=19)



# print (len(valid_generator))
num_valid_images = len(valid_generator) * VBATCH_SIZE * 20
print ('num_valid_images=' , num_valid_images)
# 
TEST_DIR ="/home_USER/AbsFngP/DNN/Specificity/HIV/data1/test/"
test_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      horizontal_flip=False,
      vertical_flip=False
     )

TBATCH_SIZE = 1
test_generator = test_datagen.flow_from_directory(TEST_DIR, 
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=TBATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode=None,
                                                  shuffle=False, seed=40)

# print (len(test_generator))
num_test_images = len(test_generator) * TBATCH_SIZE
print ('num_test_images=' , num_test_images)
# 
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
      # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

class_list = ["SITE1", "SITE2"]

FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

from keras.optimizers import SGD, Adam


NUM_EPOCHS = 30
#IMPORTANT: Save the model in filepathOut
filepathOut="./checkpoints/" + "ResNet50" +  "_model-jb1.h5"
checkpoint = ModelCheckpoint(filepathOut, monitor='val_acc', save_best_only=True, verbose=1, mode='max')
callbacks_list = [checkpoint]

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Now following this example: https://github.com/keras-team/keras/issues/9724
STEP_SIZE_TRAIN=num_train_images//train_generator.batch_size
STEP_SIZE_VALID=num_valid_images//valid_generator.batch_size

finetune_model.fit_generator(generator=train_generator, epochs=NUM_EPOCHS, workers=8,
                                       steps_per_epoch=STEP_SIZE_TRAIN,
                                       shuffle=True, callbacks=callbacks_list,
                                       validation_data=valid_generator,validation_steps=STEP_SIZE_VALID)

finetune_model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

#print('Running valid predictions with fold {}'.format(i))
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred_test=finetune_model.predict_generator(test_generator,
                                           steps=STEP_SIZE_TEST,
                                           verbose=1)


predicted_class_indices = np.argmax(pred_test,axis=1)


tlabels = (train_generator.class_indices)
tlabels = dict((v,k) for k,v in tlabels.items())
predictions = [tlabels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results-HIVr1-jb1.csv",index=False)

iprt = -1

print ('Results for "test" Set')
zro3 = [0 for z in range(len(class_list))]
kj = [zro3[:] for z in range(len(class_list))]


icnt= 0
iagree = 0
for k in range(len(predictions)):
    iprt = iprt + 1
    mypfile=ntpath.basename(filenames[k]).replace("."," ").split()
    if re.findall(predictions[k], mypfile[0]) and re.findall(mypfile[0], predictions[k]):
       print(filenames[k],' was predicted OK as ', predictions[k])
       icnt= icnt+1
       for z in range(len(class_list)):
          if re.findall(class_list[z], mypfile[0]) and re.findall(mypfile[0], class_list[z]):
             kj[z][z] = kj[z][z] + 1   # Label matches class_list[z]

    else:
       print(filenames[k],' INCORRECTLY predicted  as ', predictions[k])
       z1 = -1  
       z2 = -1   
       for z in range(len(class_list)):
          if re.findall(class_list[z], mypfile[0]) and re.findall(mypfile[0], class_list[z]):
             z1 = z   # name of png file matches class_list[z]

          if re.findall(class_list[z], predictions[k]) and re.findall(predictions[k], class_list[z]):
             z2 = z   # Wrong prediction was class_list[z]

       if z1 < 0:
          print("Error, unrecognized class of file: ", filenames[k])
       elif ( z2 < 0 ):
          print("Error, unrecognized prediction: ", predictions[k])
       elif ( z1 == z2 ):
          print("Skip, z1=z2=",z1 )
       else:
          kj[z1][z2] = kj[z1][z2] + 1   # Label matches class_list[z]

TotPrd = iprt+1

#computing generalized Kappa for 2 judges
# initialize Partial Sums
sclssA = [0 for z in range(len(class_list))]
sclssP = [0 for z in range(len(class_list))]

sclssA[0] = 0
sclssP[0] = 0
for z1 in range(len(class_list)):
    for z2 in range(len(class_list)):
      sclssA[z1] = sclssA[z1] +  kj[z1][z2]
      sclssP[z1] = sclssP[z1] +  kj[z2][z1]

sumClsA=0
for z in range(len(class_list)):
    sumClsA= sumClsA+ sclssA[z]


sumClsP=0
for z in range(len(class_list)):
    sumClsP= sumClsP+ sclssP[z]


# By Chance
ByChnce = 0
for z in range(len(class_list)):
    ByChnce = ByChnce +  sclssA[z] * sclssP[z] / TotPrd

#print ('ByChance = ', ByChnce )

iagree = 0
for z in range(len(class_list)):
    iagree = iagree + kj[z][z] 
    print ('Number of IMGs clss[',z,']=\t', sclssA[z],'\t Correctly Predicted: \t', kj[z][z], '\t Percentage: \t', kj[z][z]/sclssA[z], ' \tTotal IMGS predicted as class[',z,']= \t', sclssP[z])

print ('Total correct predictions (iagree):', iagree)


kappaGn = (iagree - ByChnce) / (TotPrd - ByChnce)

print ('Total predictions:', TotPrd )
print ('Correct predictions:', icnt, ', Incorrect predictions', TotPrd  - icnt)
print ('Percentage of correct predictions:', (icnt / TotPrd ) * 100 )
print ('kappa =', kappaGn )



print ('Computing Confusion Matrix and Class Accuracy' )

y_true = {} # true value  is given by the filename of the jpg
y_pred = {} # predictions
arr_true = [0 for z in range(len(predictions))]
arr_pred = [0 for z in range(len(predictions))]

for k in range(len(predictions)):
    mypfile=ntpath.basename(filenames[k]).replace("."," ").split()
    arr_true[k] = mypfile[0]
    arr_pred[k] = predictions[k]

y_true = list(arr_true)
y_pred = list(arr_pred)
#    print(num_list)

labels =class_list
# labels.sort()

print("Total labels: %s -> %s" % (len(labels), labels))

print("")
print("")
df = pd.DataFrame(
    data=confusion_matrix(y_true, y_pred, labels=labels),
    columns=labels,
    index=labels
)
df

#
# The following link provides additional information about the confusion matrix and accuracy:
# https://stats.stackexchange.com/questions/255465/accuracy-vs-jaccard-for-multiclass-problem

print("")
print("")
# Local (metrics per class)
#

tps = {}
fps = {}
fns = {}
precision_local = {}
recall_local = {}
f1_local = {}
accuracy_local = {}
for label in labels:
    tps[label] = df.loc[label, label]
    fps[label] = df[label].sum() - tps[label]
    fns[label] = df.loc[label].sum() - tps[label]
    tp, fp, fn = tps[label], fps[label], fns[label]

    precision_local[label] = tp / (tp + fp) if (tp + fp) > 0. else 0.
    recall_local[label] = tp / (tp + fn) if (tp + fp) > 0. else 0.
    p, r = precision_local[label], recall_local[label]

    f1_local[label] = 2. * p * r / (p + r) if (p + r) > 0. else 0.
    accuracy_local[label] = tp / (tp + fp + fn) if (tp + fp + fn) > 0. else 0.

print("#-- Local measures --#")
print("True Positives:", tps)
print("False Positives:", fps)
print("False Negatives:", fns)
print("Precision:", precision_local)
print("Recall:", recall_local)
print("F1-Score:", f1_local)
print("Accuracy:", accuracy_local)

print("")
print("")
#
# Global
#
micro_averages = {}
macro_averages = {}

correct_predictions = sum(tps.values())
den = sum(list(tps.values()) + list(fps.values()))
micro_averages["Precision"] = 1. * correct_predictions / den if den > 0. else 0.

den = sum(list(tps.values()) + list(fns.values()))
micro_averages["Recall"] = 1. * correct_predictions / den if den > 0. else 0.

micro_avg_p, micro_avg_r = micro_averages["Precision"], micro_averages["Recall"]
micro_averages["F1-score"] = 2. * micro_avg_p * micro_avg_r / (micro_avg_p + micro_avg_r) if (micro_avg_p + micro_avg_r) > 0. else 0.

macro_averages["Precision"] = np.mean(list(precision_local.values()))
macro_averages["Recall"] = np.mean(list(recall_local.values()))

macro_avg_p, macro_avg_r = macro_averages["Precision"], macro_averages["Recall"]
macro_averages["F1-Score"] = 2. * macro_avg_p * macro_avg_r / (macro_avg_p + macro_avg_r) if (macro_avg_p + macro_avg_r) > 0. else 0.

total_predictions = df.values.sum()
accuracy_global = correct_predictions / total_predictions if total_predictions > 0. else 0.

print("#-- Global measures --#")
print("Micro-Averages:", micro_averages)
print("Macro-Averages:", macro_averages)
print("Correct predictions:", correct_predictions)
print("Total predictions:", total_predictions)
print("Accuracy:", accuracy_global)

# TN
#
tns = {}
for label in set(y_true):
    tns[label] = len(y_true) - (tps[label] + fps[label] + fns[label])

print("True Negatives:", tns)

accuracy_local_new = {}
for label in labels:
    tp, fp, fn, tn = tps[label], fps[label], fns[label], tns[label]
    accuracy_local_new[label] = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0. else 0.

total_true = sum(list(tps.values()) + list(tns.values()))
total_predictions = sum(list(tps.values()) + list(tns.values()) + list(fps.values()) + list(fns.values()))
accuracy_global_new = 1. * total_true / total_predictions if total_predictions > 0. else 0.

print("")
print("Accuracy (per class), with TNs:", accuracy_local_new)
print("Accuracy (per class), without TNs:", accuracy_local)
print("Accuracy (global), with TNs:", accuracy_global_new)
print("Accuracy (global), without TNs:", accuracy_global)


from sklearn.metrics import classification_report, accuracy_score
print("Accuracy (global), using sklearn:", accuracy_score(y_true, y_pred))

print("")
print("")
print(classification_report(y_true, y_pred, digits=4))

