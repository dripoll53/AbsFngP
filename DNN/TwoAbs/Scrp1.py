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
#from datetime import datetime
# current date and time
#now = datetime.now()
#timestamp = datetime.timestamp(now)
#print("timestamp =", timestamp)

#IMPORTANT: To generate the model using ResNet50 weight filepathIn must not be used
#filepathIn=""
#IMPORTANT For predictions using an existing model, must be set to the the right file
#filepathIn="./checkpoints/" + "ResNet50" +  "_model_WEI-DTMBnoB1.h5"

HEIGHT = 150
WIDTH = 150

base_model = ResNet50(weights='imagenet', 
                       include_top=False, 
                       input_shape=(HEIGHT, WIDTH, 3))


BATCH_SIZE = 100

TRAIN_DIR = "/home_USER/AbsFngP/TwoAbs/DATAG/data1/train/"

train_datagen =  ImageDataGenerator(
       preprocessing_function=preprocess_input,
       rotation_range=75,
       horizontal_flip=False,
       vertical_flip=False
     )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=BATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode="categorical",
                                                  #class_mode="binary",
                                                  shuffle=True, seed=76)


# print (len(train_generator))
num_train_images = len(train_generator) * BATCH_SIZE  * 20
print ('num_train_images=' , num_train_images)

VALID_DIR = "/home_USER/AbsFngP/TwoAbs/DATAG/data1//valid/"

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
                                                  shuffle=True, seed=6)



#num_valid_images = 200
# print (len(valid_generator))
num_valid_images = len(valid_generator) * VBATCH_SIZE * 10
print ('num_valid_images=' , num_valid_images)
# 
# print (" dict")
# valid_generator.__dict__
# print (" keys")
# valid_generator.__dict__.keys()

TEST_DIR = "/home_USER/AbsFngP/TwoAbs/DATAG/data1//test/"
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
                                                  shuffle=False, seed=48)

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
    #IMPORTANT: To generate the model using ResNet50 weight filepathIn mus not be used
    #if(path.exists(filepathIn)):
    #     finetune_model.load_weights(filepathIn)

    return finetune_model

class_list =  ["BIND", "NBND"] 

FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

from keras.optimizers import SGD, Adam

#finetune_model = keras.models.load_model(filepathIn)
#finetune_model.summary()
NUM_EPOCHS =30
#IMPORTANT: Save the model in filepathOut
filepathOut="./checkpoints/" + "ResNet50" +  "_model-SIDj1.h5"
checkpoint = ModelCheckpoint(filepathOut, monitor='val_acc', save_best_only=True, verbose=1, mode='max')
callbacks_list = [checkpoint]

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Now following this example: https://github.com/keras-team/keras/issues/9724
#STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
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


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results-twoAbsj1.csv",index=False)

iprt = -1
icnt= 0
iiA= 0
iiB= 0
iiC= 0
iiD= 0
for k in range(len(predictions)):
    iprt = iprt + 1
    if re.search(predictions[k], filenames[k]):
       print(filenames[k],' predicted OK as ', predictions[k])
       icnt= icnt+1
       if re.search(class_list[0], filenames[k]):
            iiA =iiA + 1   # Label matches BIND
            #print ('Label matches BIND; Prediction is BIND; iiA=', iiA)
       else:
            iiD =iiD + 1   # Label matches NBND
            #print ('Label matches NBND; Prediction is NBN; iiD=', iiD)

    else:
       print(filenames[k],' INCORRECTLY predicted  as ', predictions[k])
       if re.search(class_list[0], filenames[k]):
            iiB =iiB + 1   # Label matches BIND
            #print ('Label matches BIND; Prediction is NBN; iiB=', iiB)
       else:
            iiC =iiC + 1   # Label matches NBND
            #print ('Label matches NBND; Prediction is BIN; iiC=', iiC)


print ('test Set')
print ('Total predictions:', iprt+1 )
print ('Correct predictions:', icnt, ', Incorrect predictions', (iprt+1) - icnt)
print ('Percentage of correct predictions:', (icnt / (iprt+1)) * 100 )

iTot = iiA + iiB + iiC + iiD
p0=( iiA + iiD ) / iTot
PBind= ( iiA + iiB) * (iiA + iiC) / (iTot * iTot)
PNoBind= ( iiB + iiC) * (iiB + iiD) / (iTot * iTot)
pe = PBind + PNoBind
kappa = (p0 - pe) / (1.0 - pe)

# print ('a= ', iiA, ', b= ', iiB, ', c= ', iiC, ', d= ', iiD ,' Itot=',iTot)
# print ('P(BIND)=', PBind, ', P(NBND)=', PNoBind , ', p0= ', p0 ,', pe= ', pe )
print ('kappa =', kappa )
# print ('class_list[0]=', class_list[0] )
# print ('class_list[1]=', class_list[1] )
# timestamp = datetime.timestamp(now)
# print("timestamp =", timestamp)
