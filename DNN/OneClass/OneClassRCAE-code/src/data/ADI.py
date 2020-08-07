from src.data.base import DataLoader
from src.data.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, pca
from src.utils.visualization.mosaic_plot import plot_mosaic
from src.utils.misc import flush_last_line
from src.config import Configuration as Cfg

import os
import numpy as np
import pickle
import tensorflow as tf
print(tf.__version__)
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Activation,LeakyReLU,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape,BatchNormalization, regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import SGD,Adam
from sklearn.metrics import average_precision_score,mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from skimage import io
from numpy import linalg as LA

from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.applications.resnet50 import preprocess_input
import ntpath
import re

HEIGHT = 32
WIDTH = 32
DBATCH_SIZE = 1000 
FCTRN = 20

#PROJECT_DIR = "/content/drive/My Drive/2019/testing/oc-nn/"

from keras.callbacks import Callback
# class ImageWithNames(DirectoryIterator):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.filenames_np = np.array(self.filepaths)
#         self.class_mode = None # so that we only get the images back
# 
#     def _get_batches_of_transformed_samples(self, index_array):
#         return (super()._get_batches_of_transformed_samples(index_array),
#                 self.filenames_np[index_array])

class ADI_DataLoader(DataLoader):
    mean_square_error_dict ={}
    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "adi"

        self.n_train = 45000
        self.n_val = 5000
        self.n_test = 2000
        #self.num_outliers = 500
        self.num_outliers = Cfg.NUMoutliers

        self.seed = Cfg.seed

        self.n_classes = 2

        print ("ADI_DataLoader:adi_normal is= ", Cfg.adi_normal) 

        if Cfg.ad_experiment:
            self.n_classes = 2
        else:
            self.n_classes = 10

        print("Cfg.ad_experiment:", Cfg.ad_experiment," self.n_classes:", self.n_classes)

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))
        self.data_path = Cfg.ADI_DATA_IN
        print("IMAGE Data will be loaded from:/", self.data_path)
 

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        #self.load_data()

        # print("Inside the MNIST_DataLoader RCAE.RESULT_PATH:", RCAE_AD.RESULT_PATH)
        self.rcae_results = Cfg.REPORT_OUTDIR
        self.modelsave_path = Cfg.SAVE_MODEL_DIR

        print("Inside the ADI_DataLoader RCAE.RESULT_PATH:", self.rcae_results)

        # load data from disk
        self.load_data()

        ## Rcae parameters
        self.mue = 0.1
        self.lamda = [0.01]
        self.Noise = np.zeros(len(self._X_train))
        self.anomaly_threshold = 0.0
        self.cae = self.build_autoencoder()
        self.latent_weights = [0, 0, 0]
        self.batchNo = 0
        self.index = 0
        print("After build_autoencoder")


    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu

    def load_data(self):

        from pathlib import Path
        Cfg.icnt += 1
        print ("\n\nADI_DataLoader:icnt is= ",  Cfg.icnt) 
        print("[INFO:] Loading data...")
        print("The normal label used in experiment,",Cfg.adi_normal)
        train_datagen = ImageDataGenerator(
              #preprocessing_function=preprocess_input,
              rescale=1./255,
              #rotation_range=50,
              horizontal_flip=False,
              vertical_flip=False
             )

        train_generator = train_datagen.flow_from_directory(self.data_path + "/train/",
                                                  target_size=(HEIGHT, WIDTH),
                                                  batch_size=DBATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode="categorical",
                                                  shuffle=True, seed=45) 

        print (len(train_generator))

        i = 0
        trnfilenam = []
        for filet in train_generator.filenames:
           trnfilenam.append(filet)
           i += 1

        if i < 1:
           print ("No train files found")
           quit()

        rlabels = (train_generator.class_indices)
        trlabels = dict((v,k) for k,v in rlabels.items())
        lntrlb = len(trlabels) 
        print ("trlabels=", trlabels, "length=", lntrlb)
        # generate vector with shape
        clssVec = np.arange(lntrlb)
        print ("clssVec=", clssVec)
        clsV = np.reshape(clssVec,(lntrlb,1))
        print ("clsV=", clsV)
 
        num_train_images = len(trnfilenam)  

        print ('Actual number of training images=' , num_train_images)
        print ('Batch_size for training images=' , DBATCH_SIZE,"\n")

        test_datagen =  ImageDataGenerator(
              rescale=1./255,
              horizontal_flip=False,
              vertical_flip=False
             )


        test_generator = test_datagen.flow_from_directory(self.data_path + "/test/",
                                                  target_size=(HEIGHT, WIDTH),
                                                  batch_size=DBATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode="categorical",
                                                  shuffle=True, seed=54)


        tstfilenam = []
        for files in test_generator.filenames:
           tstfilenam.append(files)
           i += 1

        if i< 1:
           print ('No test files found')
           quit()

        num_test_images = len(tstfilenam) 
        print ('Actual number of testing images=' , num_test_images)

        mytest_datagen =  ImageDataGenerator(
              rescale=1./255,
              horizontal_flip=False,
              vertical_flip=False
             )


        mytest_generator = mytest_datagen.flow_from_directory(self.data_path + "/MyTest/",
                                                  target_size=(HEIGHT, WIDTH),
                                                  batch_size=DBATCH_SIZE,
                                                  color_mode="rgb",
                                                  class_mode=None,
                                                  shuffle=False, seed=54)


        i = 0
        mytstfilenam = []
        for files in mytest_generator.filenames:
           mytstfilenam.append(files)
           i += 1

        if i< 1:
           print ('No independent test files found')
           quit()

        num_mytest_images = len(mytstfilenam) 
        print ('Actual number of mytesting images=' , num_mytest_images)

        trnfile_names = []
        trbatches_per_epoch = train_generator.samples // train_generator.batch_size + (train_generator.samples % train_generator.batch_size > 0)
        print ("train_generator.samples",train_generator.samples," trbatches_per_epoch=",trbatches_per_epoch)
        for i in range(trbatches_per_epoch):
            if i == 0:
                x_train,y_trn= train_generator.next()
            else:
                xt,yt = train_generator.next()
                x_train = np.concatenate((x_train, xt))
                y_trn = np.concatenate((y_trn, yt))

            current_index = ((train_generator.batch_index-1) * train_generator.batch_size)
            if current_index < 0:
                if train_generator.samples % train_generator.batch_size > 0:
                    current_index = max(0,train_generator.samples - train_generator.samples % train_generator.batch_size)
                else:
                    current_index = max(0,train_generator.samples - train_generator.batch_size)
            index_array = train_generator.index_array[current_index:current_index + train_generator.batch_size].tolist()
            trnf = [train_generator.filenames[idx] for idx in index_array] 
            trnfile_names = np.concatenate(( trnfile_names, trnf ) )
                    

        y_train = np.dot(y_trn.astype(int), clsV)
        mid = int(train_generator.samples/2)
        print ("1 shape y_train", y_train.shape)
        print(" y_train[mid - 5 :mid + 5]" , y_train[mid - 5:mid + 5] )
        
        
        for i in range (0, len(trnfile_names) ):
               trnfile_names[i] = Path(trnfile_names[i]).stem
               if i < 5:
                  print ("A  i= ", i, " " ,  trnfile_names[i], " y= ",y_train[i,] )    


        tstfile_names = []
        best_top10_kys = []
        tsbatches_per_epoch = test_generator.samples // test_generator.batch_size + (test_generator.samples % test_generator.batch_size > 0)
        for i in range(tsbatches_per_epoch):
            if i == 0:
                x_test,y_tst= test_generator.next()
            else:
                xt,yt = test_generator.next()
                x_test = np.concatenate((x_test, xt))
                y_tst = np.concatenate((y_tst, yt))

            #batch = next(test_generator)
            current_index = ((test_generator.batch_index-1) * test_generator.batch_size)
            if current_index < 0:
                if test_generator.samples % test_generator.batch_size > 0:
                    current_index = max(0,test_generator.samples - test_generator.samples % test_generator.batch_size)
                else:
                    current_index = max(0,test_generator.samples - test_generator.batch_size)
            index_array = test_generator.index_array[current_index:current_index + test_generator.batch_size].tolist()
            tstf = [test_generator.filenames[idx] for idx in index_array] 
            tstfile_names = np.concatenate(( tstfile_names, tstf ) )

        y_test = np.dot(y_tst.astype(int), clsV)
        print("shape x_test" , x_test.shape )
        print("shape y_test" , y_test.shape )
        print("len tstfile_names" , len(tstfile_names) )

        for i in range (0, len(tstfile_names) ):
               tstfile_names[i] = Path(tstfile_names[i]).stem
               if i < 5:
                  print ("B  i= ", i, " " ,  tstfile_names[i], " y= ",y_test[i,] )     

               if i > (len(tstfile_names) - 6):
                  print ("C  i= ", i, " " ,  tstfile_names[i], " y= ",y_test[i,] )     


        mytstfile_names = []
        best_top10_kys = []
        mytsbatches_per_epoch = mytest_generator.samples // mytest_generator.batch_size + (mytest_generator.samples % mytest_generator.batch_size > 0)
        for i in range(mytsbatches_per_epoch):
            if i == 0:
                x_mytest= mytest_generator.next()
            else:
                xt = mytest_generator.next()
                x_mytest = np.concatenate((x_mytest, xt))

            #batch = next(mytest_generator)
            current_index = ((mytest_generator.batch_index-1) * mytest_generator.batch_size)
            if current_index < 0:
                if mytest_generator.samples % mytest_generator.batch_size > 0:
                    current_index = max(0,mytest_generator.samples - mytest_generator.samples % mytest_generator.batch_size)
                else:
                    current_index = max(0,mytest_generator.samples - mytest_generator.batch_size)
            index_array = mytest_generator.index_array[current_index:current_index + mytest_generator.batch_size].tolist()
            mytstf = [mytest_generator.filenames[idx] for idx in index_array] 
            mytstfile_names = np.concatenate(( mytstfile_names, mytstf ) )



        print("shape x_mytest" , x_mytest.shape )
        print("len mytstfile_names" , len(mytstfile_names) )
        y_mytest = np.ndarray(shape=(len(mytstfile_names), 1)).astype(int)
        for i in range (0, len(mytstfile_names) ):
               mytstfile_names[i] = Path(mytstfile_names[i]).stem
               if re.match(r"S1", mytstfile_names[i]):
                  y_mytest[i] = int(0)
               elif re.match(r"S2", mytstfile_names[i]):
                  y_mytest[i] = int(1)
               else:
                  print (" no match for  i= ", i, " " ,  mytstfile_names[i])     
                  quit()
                   
               if i < 5:
                  print ("D  i= ", i, " " ,  mytstfile_names[i], " y= ",y_mytest[i,] )     

               if i > (len(mytstfile_names) - 6):
                  print ("E  i= ", i, " " ,  mytstfile_names[i], " y= ",y_mytest[i,] )     


        # normalize data
        #from keras.datasets import cifar10
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #x_train = x_train.astype('float32')
        indx1=[0]
        xx1 = np.take(x_train.shape, indx1)
        #x_test = x_test.astype('float32')
        #x_train /= 255.0
        #x_test /= 255.0
        indx2=[0]
        xx2 = np.take(x_test.shape, indx2)
        print ("\n1 shape x_test", x_test.shape)
        print ("1 shape y_test", y_test.shape)
        print ("1 tstfile_names[0:3]  ",  tstfile_names[0:3])
        print ("1 shape of tstfile_names  ",  tstfile_names.shape )

        ## Added newly
        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))
        x_filenam = np.concatenate(( trnfile_names,  tstfile_names)) 

        print ("2 shape x_train", x_train.shape)
        print ("2 shape y_train", y_train.shape)
        print ("2 x_filenam[0:3]  ",  x_filenam[0:3])
        print ("2 shape of x_filenam  ",  x_filenam.shape )
 
        print ("2 shape y_train", y_train.shape)
        
        print ("Cfg.adi_normal", Cfg.adi_normal)
        y_train = np.reshape(y_train, len(y_train))
        # generate an extra array for x_test  DRR
        x_tsst = x_test
        y_tsst = y_test

        print ("LEN y_tsst:", len(y_tsst))
        y_tsst = np.reshape(y_tsst, len(y_tsst))
        y_mytest = np.reshape(y_mytest, len(y_mytest))
        #
        #x_norm = x_train[np.where(y_train == Cfg.cifar10_normal)]
        x_norm = x_train[np.where(y_train == Cfg.adi_normal)]
        print ("3 shape x_norm", x_norm.shape)
        x_tsst_norm = x_tsst[np.where(y_tsst == Cfg.adi_normal)]
        x_mytest_norm = x_mytest[np.where(y_mytest == Cfg.adi_normal)]

        y_norm = np.zeros(len(x_norm))
        #
        y_tsst_norm = np.zeros(len(x_tsst_norm))
        y_mytest_norm = np.zeros(len(x_mytest_norm))
        #
        x_norm_fn =x_filenam[np.where(y_train == Cfg.adi_normal)]
        print (" x_norm_fn[0:4]=", x_norm_fn[0:4])
        x_tsst_norm_fn = tstfile_names[np.where(y_tsst == Cfg.adi_normal)]
        x_mytest_norm_fn =  mytstfile_names[np.where(y_mytest == Cfg.adi_normal)]


        #outliers = list(range(0, 10))
        outliers = list(range(0, 2))
        outls_tsst=list(range(0, 2))
        outls_mytest=list(range(0, 2))

        # print ("outliers:", outliers)
        #outliers.remove(Cfg.cifar10_normal)
        outliers.remove(Cfg.adi_normal)
        print ("new outliers:", outliers)
        #
        outls_tsst.remove(Cfg.adi_normal)
        outls_mytest.remove(Cfg.adi_normal)
        idx_outlier = np.any(y_train[..., None] == np.array(outliers)[None, ...], axis=1)
        print ("len idx_outlier:", len(idx_outlier) )
        print ("shape idx_outlier:", idx_outlier.shape  )
        print ("idx_outlier[0]:", idx_outlier[0]  )
        idx_outls_tsst = np.any(y_tsst[..., None] == np.array(outls_tsst)[None, ...], axis=1)
        idx_outls_mytest = np.any(y_mytest[..., None] == np.array(outls_mytest)[None, ...], axis=1)
        #######

        #DRR reduce array x_outlier to those with idx_outlier = TRUE
        x_outlier = x_train[idx_outlier]
        print ("NEW len x_outlier:", len(x_outlier) )
        x_outls_tsst= x_test[idx_outls_tsst]
        x_outls_mytest=  x_mytest[idx_outls_mytest]
        #
        x_outlier_fn = x_filenam[idx_outlier]
        print ("NEW len x_outlier_fn:", len(x_outlier_fn) )
        print (" x_outlier_fn[0:4]=", x_outlier_fn[0:4])
        x_outls_tsst_fn = tstfile_names[idx_outls_tsst]
        x_outls_mytest_fn =  mytstfile_names[idx_outls_mytest]
        #
        #############################
        # DRR: I DONT THINK THESE 4 LINE ARE NEEDED SINCE ALL y_outlier ARE SET TO 1
        y_outlier = y_train[idx_outlier]
        print (" y_outlier[0:10]=", y_outlier[0:10])
        y_outls_tsst = y_test[idx_outls_tsst]
        y_outls_mytest = y_mytest[idx_outls_mytest]
        #############################




        print("INFO: Random Seed set is ",self.seed)
        np.random.seed(self.seed)
        #x_outlier = np.random.permutation(x_outlier)[:self.num_outliers]

        # x_outlier = x_outlier[0:self.num_outliers]
        #######
        def unison_shuffled_copies(a, b, seed):
            np.random.seed(seed)
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        xp_outlier, xp_outlier_fn =  unison_shuffled_copies(x_outlier[:self.num_outliers],  \
                     x_outlier_fn[:self.num_outliers], self.seed)
        #######


        yp_outlier =  np.ones(len(xp_outlier))
        y_outls_tsst = np.ones(len(x_outls_tsst))
        y_outls_mytest = np.ones(len(x_outls_mytest))

        # xp_outlier shape
        print('xp_outlier shape:', xp_outlier.shape)
        print (" xp_outlier_fn[0:4]=", xp_outlier_fn[0:4])


        x_train = np.concatenate((x_norm, xp_outlier))
        y_train = np.concatenate((y_norm, yp_outlier))
        x_train_fn = np.concatenate((x_norm_fn, xp_outlier_fn))

        x_tsst = np.concatenate((x_tsst_norm, x_outls_tsst))
        y_tsst = np.concatenate((y_tsst_norm, y_outls_tsst))
        x_tsst_fn = np.concatenate((x_tsst_norm_fn, x_outls_tsst_fn))

        x_mytest = np.concatenate((x_mytest_norm, x_outls_mytest))
        y_mytest = np.concatenate((y_mytest_norm, y_outls_mytest))
        x_mytest_fn = np.concatenate((x_mytest_norm_fn, x_outls_mytest_fn))

        print("After RND permutation, xtrain shape:", x_train.shape)
        #for i in range (0, len(x_train_fn) ):
        #      if i > 5:
        #         break
        #       print ("After i= ", i, " " ,  x_train_fn[i], " y= ",y_train[i,] )

        self._X_train = x_train
        self._y_train = y_train
        self._X_train_fn = x_train_fn
        self._X_val = np.empty(x_train.shape)
        self._y_val = np.empty(y_train.shape)
        self._X_val_fn = np.empty(x_train_fn.shape)

        self._X_test = x_train
        self._y_test = y_train
        self._X_test_fn = x_train_fn
        #changed DRR
        self._X_tsst = x_tsst
        self._y_tsst = y_tsst
        self._X_tsst_fn = x_tsst_fn

        # Chalapatay's
        # print("INFO Saving images before gcn ....")
        self._X_test_beforegcn = x_train
        self._y_test_beforegcn = y_train

        # DRR passing "mytest" images
        self._X_mytest = x_mytest
        self._y_mytest = y_mytest
        self._X_mytest_fn =x_mytest_fn

        print("_X_test_beforegcn,",self._X_test_beforegcn.shape,np.max(self._X_test_beforegcn),\
                               np.min(self._X_test_beforegcn))

        # Xtest = Xtest/255.0

        gcn_required_for_classes = [ 3,5,6,7,9]

        # global contrast normalization
        if(Cfg.adi_normal in gcn_required_for_classes):
            if Cfg.gcn:
                [self._X_train,self._X_val,self._X_test] = global_contrast_normalization(self._X_train, \
                               self._X_val,self._X_test, scale=Cfg.unit_norm_used)
                self._X_test = self._X_train
                print('global contrast normalization for Cfg.adi_normal:',Cfg.adi_normal)

        print("Data loaded.")


        return

    def custom_rcae_loss(self):

        U = self.cae.layers[16].get_weights()
        U = U[0]

        V = self.cae.layers[19].get_weights()
        V = V[0]
        V = np.transpose(V)

        print("[INFO:] Shape of U, V",U.shape,V.shape)
        N = self.Noise
        lambda_val = self.lamda[0]
        mue = self.mue
        batch_size = 128

        # batch_size = 128
        # for index in range(0, N.shape[0], batch_size):
        #     batch = N[index:min(index + batch_size, N.shape[0]), :]
        # N_reshaped = N_reshaped[self.index:min(self.index + K.int_shape(y_true)[0], N_reshaped.shape[0]), :]
        # print("[INFO:] dynamic shape of batch is ", )
        # if(N.ndim >1):
        #
        #     N_reshaped = np.reshape(N,(len(N),32,32,3))
        #     symbolic_shape = K.shape(y_pred)
        #     noise_shape = [symbolic_shape[axis] if shape is None else shape
        #                    for axis, shape in enumerate(N_reshaped)]
        #     N_reshaped= N_reshaped[0:noise_shape[1]]
        #     term1 = keras.losses.mean_squared_error(y_true, (y_pred+N_reshaped ))
        #
        # else:
        #     term1 = keras.losses.mean_squared_error(y_true, (y_pred))

        def custom_rcae(y_true, y_pred):
            term1 = keras.losses.mean_squared_error(y_true, (y_pred))
            term2 = mue * 0.5 * (LA.norm(U) + LA.norm(V))
            term3 = lambda_val * 0.5 * LA.norm(N)
            print ( "custom_rcae:term1 ", term1.shape)
            print ( "custom_rcae:term2 ", term2.shape)
            print ( "custom_rcae:term3 ", term3.shape)
            return (term1 + term2 + term3)

        # print("[INFO:] custom_rcae = ",custom_rcae)
        return custom_rcae

    def build_autoencoder(self):

        # initialize the model
        print( "DRR build_autoencoder")
        autoencoder = Sequential()
        inputShape = (32,32,3)
        chanDim = -1 # since depth is appearing the end
        # first set of CONV => RELU => POOL layers
        autoencoder.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        autoencoder.add(Conv2D(32, (3, 3), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        autoencoder.add(Conv2D(16, (3, 3), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        autoencoder.add(Flatten())


        autoencoder.add(Dense(256))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))

        autoencoder.add(Dense(128))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))



        autoencoder.add(Dense(256))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))

        autoencoder.add(Reshape((4, 4, 16)))

        autoencoder.add(Conv2D(16, (3, 3), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(UpSampling2D(size=(2, 2)))

        autoencoder.add(Conv2D(32, (3, 3), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(UpSampling2D(size=(2, 2)))

        autoencoder.add(Conv2D(64, (3, 3), padding="same",
                               input_shape=inputShape))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(UpSampling2D(size=(2, 2)))

        autoencoder.add(Conv2D(3, (3, 3), use_bias=True, padding='same'))
        autoencoder.add(Activation('sigmoid'))

        print("[INFO:]DRR Autoencoder summary ", autoencoder.summary())

        return autoencoder


    def plot_train_history_loss(self,history):
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(self.rcae_results+"rcae_")
        plt.clf()
        plt.cla()
        plt.close()
        
        return

    def compute_mse(self,Xclean, Xdecoded, lamda):
        # print len(Xdecoded)
        Xclean = np.reshape(Xclean, (len(Xclean), 3072))
        m, n = Xclean.shape
        Xdecoded = np.reshape(np.asarray(Xdecoded), (m, n))
        # print Xdecoded.shape
        Xdecoded = np.reshape(Xdecoded, (len(Xdecoded), 3072))

        print("[INFO:] Xclean  MSE Computed shape", Xclean.shape)

        print("[INFO:]Xdecoded  Computed shape", Xdecoded.shape)

        meanSq_error = mean_squared_error(Xclean, Xdecoded)
        print("[INFO:] MSE Computed shape", meanSq_error.shape)

        ADI_DataLoader.mean_square_error_dict.update({lamda: meanSq_error})
        print("\n Mean square error Score ((Xclean, Xdecoded):")
        print(ADI_DataLoader.mean_square_error_dict.values())

        return ADI_DataLoader.mean_square_error_dict

    # Function to compute softthresholding values
    def soft_threshold(self,lamda, b):

        th = float(lamda) / 2.0
        print("(lamda,Threshold)", lamda, th)
        print("The type of b is ..., its len is ", type(b), b.shape, len(b[0]))

        if (lamda == 0):
            return b
        m, n = b.shape

        x = np.zeros((m, n))

        k = np.where(b > th)
        # print("(b > th)",k)
        # print("Number of elements -->(b > th) ",type(k))
        x[k] = b[k] - th

        k = np.where(np.absolute(b) <= th)
        # print("abs(b) <= th",k)
        # print("Number of elements -->abs(b) <= th ",len(k))
        x[k] = 0

        k = np.where(b < -th)
        # print("(b < -th )",k)
        # print("Number of elements -->(b < -th ) <= th",len(k))
        x[k] = b[k] + th
        x = x[:]

        return x

    def compute_best_worst_rank(self,testX, Xdecoded):
        # print len(Xdecoded)

        testX = np.reshape(testX, (len(testX), 3072))
        m, n = testX.shape
        Xdecoded = np.reshape(np.asarray(Xdecoded), (m, n))
        # print Xdecoded.shape
        Xdecoded = np.reshape(Xdecoded, (len(Xdecoded), 3072))

        # Rank the images by reconstruction error
        anomalies_dict = {}
        for i in range(0, len(testX)):
            anomalies_dict.update({i: np.linalg.norm(testX[i] - Xdecoded[i])})

        # Sort the recont error to get the best and worst 10 images
        best_top10_anomalies_dict = {}
        # Rank all the images rank them based on difference smallest  error
        best_sorted_keys = sorted(anomalies_dict, key=anomalies_dict.get, reverse=False)
        worst_top10_anomalies_dict = {}
        worst_sorted_keys = sorted(anomalies_dict, key=anomalies_dict.get, reverse=True)



        # Picking the top 10 images that were not reconstructed properly or badly reconstructed
        counter_best = 0
        # Show the top 10 most badly reconstructed images
        for b in best_sorted_keys:
            if (counter_best <= 29):
                counter_best = counter_best + 1
                best_top10_anomalies_dict.update({b: anomalies_dict[b]})
        best_top10_keys = best_top10_anomalies_dict.keys()

        # Picking the top 10 images that were not reconstructed properly or badly reconstructed
        counter_worst = 0
        # Show the top 10 most badly reconstructed images
        for w in worst_sorted_keys:
            if (counter_worst <= 29):
                counter_worst = counter_worst + 1
                worst_top10_anomalies_dict.update({w: anomalies_dict[w]})
        worst_top10_keys = worst_top10_anomalies_dict.keys()

        return [best_top10_keys, worst_top10_keys]

    def computePred_Labels(self, X_test, decoded, poslabelBoundary, negBoundary):

        y_pred = np.ones(len(X_test))
        recon_error = {}
        for i in range(0, len(X_test)):
            recon_error.update({i: np.linalg.norm(X_test[i] - decoded[i])})

        best_sorted_keys = sorted(recon_error, key=recon_error.get, reverse=False)
        worst_sorted_keys = sorted(recon_error, key=recon_error.get, reverse=True)
        anomaly_index = worst_sorted_keys[0:negBoundary]
        print("[INFO:] The anomaly index are ", anomaly_index)
        for key in anomaly_index:
            if (key >= poslabelBoundary):
                y_pred[key] = -1

        return y_pred

    def fit_auto_conv_AE(self,X_N,Xclean,lamda):

        print("In fit_auto_conv_AE")
        #print("[INFO:]DRR Lenet Style Autoencoder summary",autoencoder.summary())

        # DRR: To find where self.cae.compile & self.cae.predict are coming from use the commented code
        #  import inspect
        #  cc = inspect.getfile(self.cae.compile)
        #  print("[INFO] inspect.getfile for self.cae.compile", cc) 
        ##[INFO] inspect.getfile for self.cae.compile 
        ##   /..../miniconda3/envs/ml/lib/python3.6/site-packages/keras/engine/training.py
        #  cc = inspect.getfile(self.cae.predict)
        #  print("[INFO] inspect.getfile for self.cae.predict", cc) 
        ##[INFO] inspect.getfile for self.cae.predict 
        ##   /..../miniconda3/envs/ml/lib/python3.6/site-packages/keras/engine/training.py
        #  cc = inspect.getmembers(self.cae.predict)
        #  print("[INFO] inspect.getmembers for self.cae.predict", cc) 

        print("[INFO ADI)] compiling model...")
        # opt = SGD(lr=0.01, decay=0.01 / 150, momentum=0.9, nesterov=True)
        # opt =RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt =Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.cae.compile(loss=self.custom_rcae_loss(), optimizer=opt)


        self.lamda[0] = lamda
        X_N = np.reshape(X_N, (len(X_N), 32,32,3))
        print("fit_auto: length X_N =", len(X_N))
        Xclean = np.reshape(Xclean, (len(Xclean), 32,32,3))
        print("fit_auto: length Xclean =", len(Xclean))

        history = self.cae.fit(X_N, X_N,
                                          epochs = Cfg.epochIn,
                                          batch_size = Cfg.FitBatchSize, 
                                          shuffle=True,
                                          #validation_split=0.1,
                                          validation_split=Cfg.fracVal,
                                          verbose=1
                                          )
        # callbacks = self.callbacks
        self.plot_train_history_loss(history)

        # model.fit(input, Xclean, n_epoch=10,
        #           run_id="auto_encoder", batch_size=128)

        ae_output = self.cae.predict(X_N)
        #Reshape it back to 3072 pixels
        ae_output = np.reshape(ae_output, (len(ae_output), 3072))
        print("fit_auto: length ae_output =", len(ae_output))
        Xclean = np.reshape(Xclean, (len(Xclean), 3072))
        print("fit_auto:2nd.time: length Xclean =", len(Xclean))

        np_mean_mse =  np.mean(mean_squared_error(Xclean,ae_output))
        #Compute L2 norm during training and take the average of mse as threshold to set the label
        # norm = []
        # for i in range(0, len(input)):
        #      norm.append(np.linalg.norm(input[i] - ae_output[i]))
        # np_norm = np.asarray(norm)

        self.anomaly_threshold = np_mean_mse


        return ae_output

    def compute_softhreshold(self,Xtrue, N, lamda, Xclean):
 
        print("inside compute_softhreshold")
        Xtrue = np.reshape(Xtrue, (len(Xtrue), 3072))
        print("lamda passed ", lamda)
        # inner loop for softthresholding
        for i in range(0, 1):
            X_N = Xtrue - N
            print("now fit_auto_conv_AE for i=",i)
            XAuto = self.fit_auto_conv_AE(X_N,Xtrue,lamda)  
            # XAuto is the predictions on train set of autoencoder
            XAuto = np.asarray(XAuto)
            softThresholdIn = Xtrue - XAuto
            softThresholdIn = np.reshape(softThresholdIn, (len(softThresholdIn), 3072))
            N = self.soft_threshold(lamda, softThresholdIn)
            print("Number of non zero elements  for N,lamda", np.count_nonzero(N), lamda)
            print("The shape of N", N.shape)
            print("The minimum value of N ", np.amin(N))
            print("The max value of N", np.amax(N))
        self.Noise = N
        return N



    def evalPred(self,predX, trueX, trueY):

        trueX = np.reshape(trueX, (len(trueX), 3072))
        predX = np.reshape(predX, (len(predX), 3072))

        predY = np.ones(len(trueX))

        if predX.shape[1] > 1:
            # print("[INFO:] RecErr computed as (pred-actual)**2 ")
            # mse = []
            # for i in range(0, len(predX)):
            #     mse.append(mean_squared_error(trueX,predX))
            # np_mse = np.asarray(mse)
            # # print("[INFO:] The norm computed during eval")
            # # # if norm is greater than thereshold assign the value
            # predY[np.where(np_mse > self.anomaly_threshold)] = -1

            recErr = ((predX - trueX) ** 2).sum(axis=1)
        else:
            recErr = predX
            # predY = predX

        # print ("+++++++++++++++++++++++++++++++++++++++++++")
        # print (trueY)
        # print (predY)
        # print(predY.shape)
        # print(trueY.shape)
        # print ("+++++++++++++++++++++++++++++++++++++++++++")


        ap = average_precision_score(trueY, recErr)
        auc = roc_auc_score(trueY, recErr)

        prec = self.precAtK(recErr, trueY, K=10)

        return (ap, auc, prec)

    def precAtK(self,pred, trueY, K=None):

        if K is None:
            K = trueY.shape[0]

        # label top K largest predicted scores as +'ve
        idx = np.argsort(-pred)
        predLabel = -np.ones(trueY.shape)
        predLabel[idx[:K]] = 1

        # print(predLabel)

        prec = precision_score(trueY, predLabel)

        return prec

    def save_trained_model(self, model):

        ## save the model
        # serialize model to JSON
        model =  self.cae
        model_json = model.to_json()
        with open(self.modelsave_path + "DCAE_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.modelsave_path + "DCAE_wts.h5")
        print("[INFO:] Saved model to disk @ ....",self.modelsave_path)


        return

    def pretrain_autoencoder(self):
        # DRR  NOT CALLED   need 
        #             X_train_fn = self.data._X_train_fn 
        #
        print("[INFO:] Pretraining Autoencoder start...")
        X_train = np.concatenate((self.data._X_train,self.data._X_val))
        y_train = np.concatenate((self.data._y_train, self.data._y_val))

        trainXPos = X_train[np.where(y_train == 1)]
        trainYPos = np.ones(len(trainXPos))
        trainXNeg = X_train[np.where(y_train == -1)]
        trainYNeg = -1*np.ones(len(trainXNeg))

        PosBoundary = len(trainXPos)
        NegBoundary = len(trainXNeg)


        # print("[INFO:]  Length of Positive data",len(trainXPos))
        # print("[INFO:]  Length of Negative data", len(trainXNeg))


        X_train = np.concatenate((trainXPos,trainXNeg))
        y_train = np.concatenate((trainYPos,trainYNeg))


        X_test = X_train
        y_test = y_train

        # X_test = self.data._X_test
        # y_test = self.data._y_test

        # define lamda set
        lamda_set = [ 0.1]
        mue = 0.0
        TRIALS = 1
        ap = np.zeros((TRIALS,))
        auc = np.zeros((TRIALS,))
        prec = np.zeros((TRIALS,))
        # outer loop for lamda
        for l in range(0, len(lamda_set)):
            # Learn the N using softthresholding technique
            N = 0
            lamda = lamda_set[l]
            XTrue = X_train
            YTrue = y_train

            # Capture the structural Noise
            self.compute_softhreshold(XTrue, N, lamda, XTrue)
            N = self.Noise
            # Predict the conv_AE autoencoder output
            # XTrue = np.reshape(XTrue, (len(XTrue), 32,32,3))

            print("pretrain_autoencoder: DRR calling self.cae.predict")
            print("pretrain_autoencoder: length X_test:",len(X_test))
            decoded = self.cae.predict(X_test)
          

            # compute MeanSqared error metric
            self.compute_mse(X_test, decoded, lamda)

            # rank the best and worst reconstructed images
            [best_top10_keys, worst_top10_keys] = self.compute_best_worst_rank(X_test, decoded)

            # Visualise the best and worst ( image, BG-image, FG-Image)
            # XPred = np.reshape(np.asarray(decoded), (len(decoded), 32,32,3))
            #
            # DRR
            self.visualise_anomalies_detected(    X_test,     X_test, decoded, N,  XTruefn     ,  best_top10_keys, worst_top10_keys, lamda)

            XPred = decoded

            y_pred = self.computePred_Labels(X_test,decoded,PosBoundary,NegBoundary)


            # (ap[l], auc[l], prec[l]) = self.nn_model.evalPred(XPred, X_test, y_test)
            auc[l] = roc_auc_score(y_test, y_pred)

            # print("AUPRC", lamda, ap[l])
            # print("AUROC", lamda, auc[l])
            # print("P@10", lamda, prec[l])
            print("=====================")
            print("AUROC", lamda, auc[l])
            print("=======================")
            print("[INFO:] Pretraining Autoencoder end saving autoencoder model @...")
            print("[INFO] serializing network and saving trained weights...")
            print("[INFO] Saving model config and layer weights...")
            self.save_trained_model()


        # print('AUPRC = %1.4f +- %1.4f' % (np.mean(ap), np.std(ap) / np.sqrt(TRIALS)))
        # print('AUROC = %1.4f +- %1.4f' % (np.mean(auc), np.std(auc) / np.sqrt(TRIALS)))
        # print('P@10  = %1.4f +- %1.4f' % (np.mean(prec), np.std(prec) / np.sqrt(TRIALS)))



        # print("\n Mean square error Score ((Xclean, Xdecoded):")
        # print(MNIST_DataLoader.mean_square_error_dict.values())
        # for k, v in MNIST_DataLoader.mean_square_error_dict.items():
        #     print(k, v)
        # # basic plot
        # data = MNIST_DataLoader.mean_square_error_dict.values()

        return

    def visualise_anomalies_detected(self,testX, noisytestX, decoded, N, testXNm, best_top10_keys, worst_top10_keys, lamda):

        import numpy as np
        import matplotlib.pyplot as plt

        side =32
        channel = 3
        N = np.reshape(N, (len(N), 32, 32, 3))

        best_top10_keys = list(best_top10_keys)

        imgpil = np.ndarray(shape=(10,side, side, channel))
        imgNam = ["" for i in range(10)]
        for j in range(0, 1):
            if j > 0:
               break    
            img = np.ndarray(shape=(side , side , channel))
            for k in range(0, 10):
               i= 10 * j + k
                
               if j == 0:
                  img = testX[best_top10_keys[k]] 
               elif j == 1:
                  img = noisytestX[best_top10_keys[k]] 
               elif j == 2:
                  img = decoded[best_top10_keys[k]] 
               elif j == 3:
                  img = N[best_top10_keys[k]] 

               img = 255 * img 
               imgpil[i,:,:,:] =  img
               if len(testXNm[best_top10_keys[k]]) > 15:
                    first_chars = testXNm[best_top10_keys[k]][0:7]
                    last_chars = testXNm[best_top10_keys[k]][-6:]
                    imgNam[i] = str(first_chars) + str(last_chars)
               else:
                    imgNam[i] = testXNm[best_top10_keys[k]]

        fig, axes = plt.subplots(1,10,figsize=(8,4))

# The issue with color may be relate to the "astype('uint8')" that was missing below,
#  or to the figure padding I used earlier.
 

        for i,ax in enumerate(axes.flat):
            ax.imshow(imgpil[i,:,:,:].astype('uint8'))
            ax.set_title(imgNam[i], fontsize=5)

            ax.set_yticklabels([])    # no y labels
            ax.set_xticklabels([])    # no x labels
            ax.set_yticks([])         # no y sticks
            ax.set_xticks([])         # no x sticks

        fig.tight_layout()
        #fig.tight_layout(pad=1.0)

        print("\nSaving (ADI.py) results for best after being encoded and decoded: @")
        print(self.rcae_results + '/best/')
        plt.savefig(self.rcae_results + '/best/' + str(lamda) + '_RCAEbestFig.png', dpi=150)
        plt.clf()
        plt.cla()
        plt.close()


        worst_top10_keys = list(worst_top10_keys)
        imgpil = np.ndarray(shape=(10,side, side, channel))
        imgNam = ["" for i in range(10)]
        #Only write for testX
        for j in range(0, 1):
            if j > 0:
               break    
            img = np.ndarray(shape=(side , side , channel))
            for k in range(0, 10):
               i= 10 * j + k
                
               if j == 0:
                  img = testX[worst_top10_keys[k]] 
               elif j == 1:
                  img = noisytestX[worst_top10_keys[k]] 
               elif j == 2:
                  img = decoded[worst_top10_keys[k]] 
               elif j == 3:
                  img = N[worst_top10_keys[k]] 

               img = 255 * img 
               imgpil[i,:,:,:] =  img
               if len(testXNm[worst_top10_keys[k]]) > 15:
                    first_chars = testXNm[worst_top10_keys[k]][0:7]
                    last_chars = testXNm[worst_top10_keys[k]][-6:]
                    imgNam[i] = str(first_chars) + str(last_chars)
               else:
                    imgNam[i] = testXNm[worst_top10_keys[k]]

        fig, axes = plt.subplots(1,10,figsize=(8,4))
        #fig, axes = plt.subplots(4,10,figsize=(8,4))

# The issue with color may be relate to the "astype('uint8')" that was missing below,
#  or to the figure padding I used earlier.
 

        for i,ax in enumerate(axes.flat):
            ax.imshow(imgpil[i,:,:,:].astype('uint8'))
            ax.set_title(imgNam[i], fontsize=5)

            ax.set_yticklabels([])    # no y labels
            ax.set_xticklabels([])    # no x labels
            ax.set_yticks([])         # no y sticks
            ax.set_xticks([])         # no x sticks

        fig.tight_layout()
        #fig.tight_layout(pad=1.0)

        print("\nSaving (ADI.py) results for worst after being encoded and decoded: @")
        print(self.rcae_results + '/worst/')
        plt.savefig(self.rcae_results + '/worst/' + str(lamda) + '_RCAEwrstFig.png', dpi=150)
        plt.clf()
        plt.cla()
        plt.close()

        return

    def Xile_raster_visualise_anomalies_detected(self, testX, testX_fn, worst_top10_keys, TitleStrng): 

        import matplotlib.pyplot as plt
        nrows=10 
        ncols=10
        #if len(testX)/10 > 10:
        #    ncols=10
        #side = self.IMG_HGT
        side = 32
        #channel = self.channel
        channel = 3

        worst_top10_keys = list(worst_top10_keys[1:100])
        maxit = min (99,len(worst_top10_keys))   
        imgpil = np.ndarray(shape=((nrows * ncols),side, side, channel))
        imgNam = ["" for i in range((nrows * ncols))]
        is_looping = True
        for j in range(0, nrows):
            img = np.ndarray(shape=(side , side , channel))
            for k in range(0, ncols):
               i= nrows * j + k
               #if i >=  len(testX):
               if i >= maxit:
                    is_looping = False
                    break # break out of the inner loop
                
               img = testX[worst_top10_keys[i]]
               img = 255 * img
               imgpil[i,:,:,:] =  img
               if len(testX_fn[worst_top10_keys[i]] ) > 15:
                    first_chars = testX_fn[worst_top10_keys[i]][0:6]
                    last_chars =  testX_fn[worst_top10_keys[i]][-4:]
                    imgNam[i] = str(i) + str('-') + str(first_chars) + str(last_chars)
               else:
                    imgNam[i] =  str(i) + str('-') + testX_fn[worst_top10_keys[i]]

            if not is_looping:
               break # break out of outer loop         


        fig, axes = plt.subplots(nrows,ncols,figsize=(8,8))

# The issue with color may be relate to the "astype('uint8')" that was missing below,
#  or to the figure padding I used earlier.


        for i,ax in enumerate(axes.flat):
            ax.imshow(imgpil[i,:,:,:].astype('uint8'))
            ax.set_title(imgNam[i], fontsize=5)

            ax.set_yticklabels([])    # no y labels
            ax.set_xticklabels([])    # no x labels
            ax.set_yticks([])         # no y sticks
            ax.set_xticks([])         # no x sticks

        fig.tight_layout()
        fig.tight_layout(pad=1.0)

        if(self.dataset_name == "dogs"):
           mystrn =  self.rcae_results + TitleStrng  + '_Class_'+ str(Cfg.dogs_normal) + ".png"
        if(self.dataset_name == "adi"):
           mystrn =  self.rcae_results + TitleStrng  + '_Class_'+ str(Cfg.adi_normal) + ".png"
        print("plot mystrn= ",mystrn)
        plt.savefig(str(mystrn), dpi=150)
        plt.clf()
        plt.cla()
        plt.close()


        return 


