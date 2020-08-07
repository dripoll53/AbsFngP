#   THIS SCRIPT WORKS OK 
import os.path
from os import path
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.callbacks
#from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import re
import pandas as pd
import ntpath
#from datetime import datetime
# current date and time
#now = datetime.now()
#timestamp = datetime.timestamp(now)
#print("timestamp =", timestamp)


HEIGHT = 150
WIDTH = 150

base_model = ResNet50(weights='imagenet', 
                       include_top=False, 
                       input_shape=(HEIGHT, WIDTH, 3))


BATCH_SIZE = 100

TRAIN_DIR = "/home_USER/AbsFngP/DNN/Lineage/DATAG/data1/train/"

train_datagen =  ImageDataGenerator(
       preprocessing_function=preprocess_input,
       rotation_range=50,
       horizontal_flip=False,
       vertical_flip=False
     )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=BATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode="categorical",
                                                  #class_mode="binary",
                                                  shuffle=True, seed=51)


# print (len(train_generator))
num_train_images = len(train_generator) * BATCH_SIZE  * 30
print ('num_train_images=' , num_train_images)

VALID_DIR = "/home_USER/AbsFngP/DNN/Lineage/DATAG/data1/valid/"

valid_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      #rotation_range=90,
      horizontal_flip=False,
      vertical_flip=False
     )

VBATCH_SIZE = BATCH_SIZE 
valid_generator = valid_datagen.flow_from_directory(VALID_DIR, 
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=VBATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode="categorical",
                                                  #class_mode="binary",
                                                  shuffle=True, seed=88)



#num_valid_images = 200
# print (len(valid_generator))
num_valid_images = len(valid_generator) * VBATCH_SIZE * 10
print ('num_valid_images=' , num_valid_images)
# 
# print (" dict")
# valid_generator.__dict__
# print (" keys")
# valid_generator.__dict__.keys()

TEST_DIR = "/home_USER/AbsFngP/DNN/Lineage/DATAG/data1/test/"
test_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      #rotation_range=90,
      horizontal_flip=False,
      vertical_flip=False
     )

TBATCH_SIZE = 1
test_generator = test_datagen.flow_from_directory(TEST_DIR, 
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=TBATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode=None,
                                                  shuffle=False, seed=4)

#num_test_images = 200
# print (len(test_generator))
num_test_images = len(test_generator) * TBATCH_SIZE
print ('num_test_images=' , num_test_images)
# 
# print (" dict")
# test_generator.__dict__
# print (" keys")
# test_generator.__dict__.keys()


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

class_list = ["SITE1", "SITE2", "SITE3", "SITE4", "SITE5", "SITE6", "SITE7", "SITE8", "SITE9", "SITE10"]

FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

from keras.optimizers import SGD, Adam

#finetune_model = keras.models.load_model(filepathIn)
#finetune_model.summary()
#Number of itertions

NUM_EPOCHS = 100
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

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred_test=finetune_model.predict_generator(test_generator,
                                           steps=STEP_SIZE_TEST,
                                           verbose=1)


predicted_class_indices = np.argmax(pred_test,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results-A10r1Ejb1.csv",index=False)

iprt = -1

print ('Results for "test" Set')
zro3 = [0 for z in range(len(class_list))]
kj = [zro3[:] for z in range(len(class_list))]


icnt= 0
iagree = 0
for k in range(len(predictions)):
    iprt = iprt + 1
    mypfile=ntpath.basename(filenames[k]).replace("."," ").split()
    # print ("mypfile= ", mypfile, " mypfile0=", mypfile[0])
    if re.findall(predictions[k], mypfile[0]) and re.findall(mypfile[0], predictions[k]):
       #print ("(findall) Predictions is:", predictions[k]," filename is: ",   mypfile[0])
       print(filenames[k],' was predicted OK as ', predictions[k])
       icnt= icnt+1
       for z in range(len(class_list)):
          if re.findall(class_list[z], mypfile[0]) and re.findall(mypfile[0], class_list[z]):
             kj[z][z] = kj[z][z] + 1   # Label matches class_list[z]
             #print ('Label matches ',class_list[z],' - Prediction is ',predictions[k],' - kj[',z,'][',z,'] =', kj[z][z])

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
          #print ("kj[",z1,"][", z2,"]= ",kj[z1][z2] )

TotPrd = iprt+1

#computing generalized Kappa for 2 judges
iagree = 0
for z in range(len(class_list)):
    iagree = iagree + kj[z][z] 
    print ('Correct Preds. for clss[',z,']= ',kj[z][z])

print ('Total correct predictions (iagree):', iagree)

# initialize Partial Sums
sclssA = [0 for z in range(len(class_list))]
sclssP = [0 for z in range(len(class_list))]

sclssA[0] = 0
sclssP[0] = 0
for z1 in range(len(class_list)):
    for z2 in range(len(class_list)):
      #print ('kj [',z1,'][',z2,']=',  kj[z1][z2],'  kj [',z2,'][',z1,']=',  kj[z2][z1]) 
      sclssA[z1] = sclssA[z1] +  kj[z1][z2]
      sclssP[z1] = sclssP[z1] +  kj[z2][z1]

sumClsA=0
for z in range(len(class_list)):
    print ('sclssA[',z,']=',sclssA[z])
    sumClsA= sumClsA+ sclssA[z]

print ('sum ROWS=',sumClsA)

sumClsP=0
for z in range(len(class_list)):
    print ('sclssP[',z,']=',sclssP[z])
    sumClsP= sumClsP+ sclssP[z]

print ('sum COLUMNS=',sumClsP)

# By Chance
ByChnce = 0
for z in range(len(class_list)):
    ByChnce = ByChnce +  sclssA[z] * sclssP[z] / TotPrd

print ('ByChance = ', ByChnce )

kappaGn = (iagree - ByChnce) / (TotPrd - ByChnce)


print ('Total predictions:', TotPrd )
print ('Correct predictions:', icnt, ', Incorrect predictions', TotPrd  - icnt)
print ('Percentage of correct predictions:', (icnt / TotPrd ) * 100 )
print ('kappa =', kappaGn )
