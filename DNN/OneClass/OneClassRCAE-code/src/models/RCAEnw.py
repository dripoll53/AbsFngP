# import the necessary packages
import numpy as np
from src.data.preprocessing import learn_dictionary
from sklearn.metrics import average_precision_score,mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from skimage import io

from src.config import Configuration as Cfg

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
import tensorflow as tf

sess = tf.Session()


from keras import backend as K
K.set_session(sess)

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.python.keras.callbacks import ModelCheckpoint
#
## Set the config values 
#config = tf.ConfigProto(intra_op_parallelism_threads=<NUM_PARALLEL_EXEC_UNITS>, 
#inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': 
#<NUM_PARALLEL_EXEC_UNITS> })
#
##Create the session
#session = tf.Session(config=config)
#tf.keras.backend.set_session(session)

from src.data.main import load_dataset

#PROJECT_DIR = "/content/drive/My Drive/2019/testing/oc-nn/"

class RCAE_AD:
    ## Initialise static variables
    INPUT_DIM = 0
    HIDDEN_SIZE = 0
    DATASET = "mnist"
    mean_square_error_dict = {}
    RESULT_PATH = ""

    def __init__(self, dataset, inputdim, hiddenLayerSize, img_hgt, img_wdt,img_channel, modelSavePath, reportSavePath,
                 preTrainedWtPath, seed, intValue=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        RCAE_AD.DATASET = dataset.lower()
        RCAE_AD.INPUT_DIM = inputdim
        RCAE_AD.HIDDEN_SIZE = hiddenLayerSize
        RCAE_AD.RESULT_PATH = reportSavePath
        RCAE_AD.RANDOM_SEED = seed

        print("RCAE.RESULT_PATH:",RCAE_AD.RESULT_PATH)
        print("RCAE_AD.DATASET:",RCAE_AD.DATASET)
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam

        self.directory = modelSavePath
        self.results = reportSavePath
        self.pretrainedWts = preTrainedWtPath
        self.model = ""

        self.IMG_HGT = img_hgt
        self.IMG_WDT = img_wdt
        self.channel = img_channel
        self.h_size = RCAE_AD.HIDDEN_SIZE
        global model
        self.r = 1.0
        self.kvar = 0.0
        self.pretrain= True
        self.dataset = dataset.lower()
        
        print("INFO: The load_dataset is ", self.dataset)

        # load dataset
        load_dataset(self, dataset.lower(), self.pretrain)

        print("RCAE_AD.DATASET initial:",RCAE_AD.DATASET)
   
        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "mnist"):
            from src.data.mnist import MNIST_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = MNIST_DataLoader()
            #self.mnist_savedModelPath= PROJECT_DIR+"/models/MNIST/RCAE/"
            self.mnist_savedModelPath= Cfg.SAVE_MODEL_DIR


        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "cifar10"):
            # ++++++++++++
            from src.data.cifar10 import CIFAR_10_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = CIFAR_10_DataLoader()


        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "gtsrb"):
            from src.data.GTSRB import GTSRB_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = GTSRB_DataLoader()

        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "dogs"):
            from src.data.DOGS import DOGS_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = DOGS_DataLoader()

        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "catsdogs"):
            from src.data.cDOGS import cDOGS_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = cDOGS_DataLoader()

        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "adi"):
            from src.data.ADI import ADI_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = ADI_DataLoader()

    
    def save_model(self, model,lambdaval):

        ## save the model
        # serialize model to JSON
        if(RCAE_AD.DATASET == "mnist"):
            model_json = model.to_json()
            with open(self.mnist_savedModelPath +lambdaval+ "__DCAE_DIGIT__"+str(Cfg.mnist_normal)+"__model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(self.mnist_savedModelPath + lambdaval+"__DCAE_DIGIT__"+str(Cfg.mnist_normal)+"__model.h5")
            print("Saved model to disk....")


        return

    def computePred_Labels(self, X_test,decoded,poslabelBoundary,negBoundary,lamda):

        y_pred = np.ones(len(X_test))
        recon_error = {}
        for i in range(0, len(X_test)):
            recon_error.update({i: np.linalg.norm(X_test[i] - decoded[i])})

        best_sorted_keys = sorted(recon_error, key=recon_error.get, reverse=False)
        worst_sorted_keys = sorted(recon_error, key=recon_error.get, reverse=True)
        anomaly_index = worst_sorted_keys[0:negBoundary]
        print("[INFO:] The anomaly index are ",anomaly_index)
        
       
        worstreconstructed_Top200index = worst_sorted_keys[0:200]
        print("[INFO:] The worstreconstructed_Top200index index are ",worstreconstructed_Top200index)
        
        
        
        for key in anomaly_index:
            if(key >= poslabelBoundary):
                y_pred[key] = -1
        
        top_100_anomalies= []
        for i in worstreconstructed_Top200index:
            top_100_anomalies.append(X_test[i])

        top_100_anomalies = np.asarray(top_100_anomalies)

        top_100_anomalies = np.reshape(top_100_anomalies,(len(top_100_anomalies),28,28))
        
        result = self.tile_raster_images(top_100_anomalies, [28, 28], [10, 10])
        print("[INFO:] Saving Anomalies Found at ..",self.results)
        io.imsave(self.results  + str(lamda)+"_Top100_anomalies.png",result)



        return y_pred

    def load_data(self, data_loader=None, pretrain=False):
        print("RCAEnw.py: load_data ")
        self.data = data_loader()
        return
    
    def scale_to_unit_interval(self,ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    def tile_raster_images(self,X, img_shape, tile_shape, tile_spacing=(0, 0),
                           scale_rows_to_unit_interval=True,
                           output_pixel_vals=True):
        """
        Source : http://deeplearning.net/tutorial/utilities.html#how-to-plot
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.
        """

        assert len(img_shape) == 2
        assert len(tile_shape) == 2
        assert len(tile_spacing) == 2

        # The expression below can be re-written in a more C style as
        # follows :
        #
        # out_shape = [0,0]
        # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
        #                tile_spacing[0]
        # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
        #                tile_spacing[1]
        out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                     in zip(img_shape, tile_shape, tile_spacing)]

        if isinstance(X, tuple):
            assert len(X) == 4
            # Create an output numpy ndarray to store the image
            if output_pixel_vals:
                out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
            else:
                out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

            # colors default to 0, alpha defaults to 1 (opaque)
            if output_pixel_vals:
                channel_defaults = [0, 0, 0, 255]
            else:
                channel_defaults = [0., 0., 0., 1.]

            for i in range(4):
                if X[i] is None:
                    # if channel is None, fill it with zeros of the correct
                    # dtype
                    out_array[:, :, i] = np.zeros(out_shape,
                                                  dtype='uint8' if output_pixel_vals else out_array.dtype
                                                  ) + channel_defaults[i]
                else:
                    # use a recurrent call to compute the channel and store it
                    # in the output
                    out_array[:, :, i] = self.tile_raster_images(X[i], img_shape, tile_shape,
                                                            tile_spacing, scale_rows_to_unit_interval,
                                                            output_pixel_vals)
            return out_array

        else:
            # if we are dealing with only one channel
            H, W = img_shape
            Hs, Ws = tile_spacing

            # generate a matrix to store the output
            out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

            for tile_row in range(tile_shape[0]):
                for tile_col in range(tile_shape[1]):
                    if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                        if scale_rows_to_unit_interval:
                            # if we should scale values to be between 0 and 1
                            # do this by calling the `scale_to_unit_interval`
                            # function
                            this_img = self.scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                        else:
                            this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                            # add the slice to the corresponding position in the
                            # output array
                        out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] \
                            = this_img * (255 if output_pixel_vals else 1)
            return out_array

    def get_oneClass_testData(self):

        X_train = self.data._X_train
        y_train = self.data._y_train
        
        X_test = self.data._X_test
        y_test = self.data._y_test

        
        ## Combine the positive data
        trainXPos = X_train[np.where(y_train == 0)]
        trainYPos = np.zeros(len(trainXPos))
        
        testXPos = X_test[np.where(y_test == 0)]
        testYPos = np.zeros(len(testXPos))
        
        
        # Combine the negative data
        trainXNeg = X_train[np.where(y_train == 1)]
        trainYNeg = np.ones(len(trainXNeg))
        
        testXNeg = X_test[np.where(y_test == 1)]
        testYNeg = np.ones(len(testXNeg))

     
        X_testPOS = np.concatenate((trainXPos, testXPos))
        y_testPOS = np.concatenate((trainYPos, testYPos))
        
        X_testNEG = np.concatenate((trainXNeg, testXNeg))
        y_testNEG = np.concatenate((trainYNeg, testYNeg))
        
        # Just 0.01 points are the number of anomalies.
        
        # Just 0.01 points are the number of anomalies
        if(self.dataset == "mnist"):
            num_of_anomalies = int(0.01 * len(X_testPOS))
        elif(self.dataset == "cifar10"):
            num_of_anomalies = int(0.1 * len(X_testPOS))
        elif((self.dataset == "dogs") or (self.dataset == "catsdogs")or (self.dataset == "adi")):
            num_of_anomalies = int(0.1 * len(X_testPOS))

        X_testNEG = X_testNEG[0:num_of_anomalies]
        y_testNEG = y_testNEG[0:num_of_anomalies]
        
        
        X_test = np.concatenate((X_testPOS, X_testNEG))
        y_test = np.concatenate((y_testPOS, y_testNEG))
        
        
        PosBoundary= len(X_testPOS)
        NegBoundary = len(X_testNEG)


        print("[INFO: ] Shape of One Class Input Data used in testing", X_test.shape)
        print("[INFO: ] Shape of (Positive) One Class Input Data used in testing", X_testPOS.shape)
        print("[INFO: ] Shape of (Negative) One Class Input Data used in testing", X_testNEG.shape)

        return [X_test, y_test]

    def get_oneClass_trainData(self):
        # X_train = np.concatenate((self.data._X_train, self.data._X_val))
        # y_train = np.concatenate((self.data._y_train, self.data._y_val))

        X_train = self.data._X_train
        y_train = self.data._y_train
        
        X_test = self.data._X_test
        y_test = self.data._y_test

        
        ## Combine the positive data
        trainXPos = X_train[np.where(y_train == 0)]
        trainYPos = np.zeros(len(trainXPos))
        
        testXPos = X_test[np.where(y_test == 0)]
        testYPos = np.zeros(len(testXPos))
        
        
        # Combine the negative data
        trainXNeg = X_train[np.where(y_train == 1)]
        trainYNeg = np.ones(len(trainXNeg))
        
        testXNeg = X_test[np.where(y_test == 1)]
        testYNeg = np.ones(len(testXNeg))

     
        X_trainPOS = np.concatenate((trainXPos, testXPos))
        y_trainPOS = np.concatenate((trainYPos, testYPos))
        
        X_trainNEG = np.concatenate((trainXNeg, testXNeg))
        y_trainNEG = np.concatenate((trainYNeg, testYNeg))
        
        # Just 0.01 points are the number of anomalies.
        if(self.dataset == "mnist"):
            num_of_anomalies = int(0.01 * len(X_trainPOS))
        elif(self.dataset == "cifar10"):
            num_of_anomalies = int(0.1 * len(X_trainPOS))
        elif(self.dataset == "gtsrb"):
            num_of_anomalies = int(0.1 * len(X_trainPOS))
        elif((self.dataset == "dogs") or (self.dataset == "catsdogs")or (self.dataset == "adi")):
            num_of_anomalies = int(0.1 * len(X_trainPOS))

        X_trainNEG = X_trainNEG[0:num_of_anomalies]
        y_trainNEG = y_trainNEG[0:num_of_anomalies]
        
        
        X_train = np.concatenate((X_trainPOS, X_trainNEG))
        y_train = np.concatenate((y_trainPOS, y_trainNEG))
        
    
        print("[INFO: ] Shape of One Class Input Data used in training", X_train.shape)
        print("[INFO: ] Shape of (Positive) One Class Input Data used in training", X_trainPOS.shape)
        print("[INFO: ] Shape of (Negative) One Class Input Data used in training", X_trainNEG.shape)

        return [X_train,y_train]


    def old_tile_raster_visualise_anomalies_detected(self, testX, worst_top10_keys, lamda, nrows=10, ncols=10):
        #
        # print("[INFO:] The shape of input data  ",testX.shape)
        # print("[INFO:] The shape of decoded  data  ", decoded.shape)


        side = self.IMG_HGT
        channel = self.channel
        # side = 28
        # channel = 1
        # Display the decoded Original, noisy, reconstructed images


        img = np.ndarray(shape=(side * nrows, side * ncols, channel))
        print("img shape:", img.shape)

        worst_top10_keys = list(worst_top10_keys)

        # Display the decoded Original, noisy, reconstructed images for worst
        img = np.ndarray(shape=(side * nrows, side * ncols, channel))
        for i in range(ncols):
            row = i // ncols * nrows
            col = i % ncols
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+2*ncols]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+3*ncols]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+4*ncols]]
            img[side * (row + 4):side * (row + 5), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+5*ncols]]
            img[side * (row + 5):side * (row + 6), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+6*ncols]]
            img[side * (row + 6):side * (row + 7), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+7*ncols]]
            img[side * (row + 7):side * (row + 8), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+8*ncols]]
            img[side * (row + 8):side * (row + 9), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+9*ncols]]
            img[side * (row + 9):side * (row + 10), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+10*ncols]]

        img *= 255
        img = img.astype(np.uint8)

        # # Save the image decoded
        if(self.dataset == "cifar10"):
            # Save the image decoded
            print('Saving '+ str(lamda) + " most anomalous digit: @")
            io.imsave(self.results + str(lamda)  + 'Class_'+ str(Cfg.cifar10_normal) + "_Top100.png", img)
        if(self.dataset == "mnist"):
            # Save the image decoded
            img = np.reshape(img, (side * nrows, side * ncols))
            print('Saving '+ str(lamda) + " most anomalous digit: @")
            io.imsave(self.results + str(lamda)  + 'Class_'+ str(Cfg.mnist_normal) + "_Top100.png", img)
        if((self.dataset == "dogs") or (self.dataset == "catsdogs")):
            # Save the image decoded
            print('Saving '+ str(lamda) + " most anomalous digit: @")
            io.imsave(self.results + str(lamda)  + 'Class_'+ str(Cfg.dogs_normal) + "_Top100.png", img)
        if(self.dataset == "adi"):
            # Save the image decoded
            print('Saving '+ str(lamda) + " most anomalous digit: @")
            io.imsave(self.results + str(lamda)  + 'Class_'+ str(Cfg.adi_normal) + "_Top100.png", img)

        return
    

    #def save_Most_Normal(self,X_test,scores):
    def save_Most_Normal(self,X_test,X_test_fn,k,p, scores):
        worst_sorted_keys= np.argsort(scores)
        print(" len(worst_sorted_keys)= ",len(worst_sorted_keys) ) 
        print("worst_sorted_keys.shape= ",len(worst_sorted_keys.shape )) 
        most_anomalous_index = worst_sorted_keys[0:100]
        top_100_anomalies = []
        # DRR to be added 
        #top_100_anomalies_fn= []

        for i in most_anomalous_index:
            if i< 100:
              top_100_anomalies.append(X_test[i])
              # DRR to be added 
              #top_100_anomalies_fn.append(X_test[i])
            else:
              break

        top_100_anomalies = np.asarray(top_100_anomalies)
        # DRR to be added 
        #top_100_anomalies_fn = np.asarray(top_100_anomalies_fn)
        
        if p == 0:
           titlStr  = str("most_normal-") + str(k) 
        elif p==2:
           titlStr  = str("t2-most_normal-") + str(k) + str("-")  + str(p)
        else:
           titlStr  = str("t3-most_normal-") + str(k) + str("-")   + str(p)

        print("Normal file name string=",titlStr) 
        print("[INFO:] The top_100_normal iter.(",k,") ",top_100_anomalies.shape)
        print("[INFO:] The top_100_normal iter.(",k,") -- len= ",len(top_100_anomalies))
        self.nn_model.Xile_raster_visualise_anomalies_detected(X_test,X_test_fn,worst_sorted_keys[1:100],titlStr)

        return 
     
    #def save_Most_Anomalous(self,X_test,scores):
    def save_Most_Anomalous(self,X_test,X_test_fn,k,p, scores):

        worst_sorted_keys= np.argsort(-scores)
        most_anomalous_index = worst_sorted_keys[0:100]
        top_100_anomalies = []

        for i in most_anomalous_index:
            if i< 100:
              top_100_anomalies.append(X_test[i])
            else:
              break

        top_100_anomalies = np.asarray(top_100_anomalies)
        if p == 0:
           titlStr  = str("most_anomalous-") + str(k) 
        elif p==2:
           titlStr  = str("t2-most_anomalous-") + str(k) + str("-")  + str(p)
        else:
           titlStr  = str("t3-most_anomalous-") + str(k) + str("-")  + str(p)

        print("Anomalous filename string=",titlStr) 
        print("[INFO:] The top_100_anomalies iter.(",k,") ",top_100_anomalies.shape)
        print("[INFO:] The top_100_anomalies iter.(",k,") -- len= ",len(top_100_anomalies))
        self.nn_model.Xile_raster_visualise_anomalies_detected(X_test,X_test_fn,worst_sorted_keys[1:100],titlStr)

        return 

    def fit_and_predict(self):

        print("(f&p)RCAE_AD.DATASET:",RCAE_AD.DATASET)
        print("(f&p)self.dataset:", self.dataset)
      
        if(self.dataset == "cifar10"):
            X_train = self.data._X_train
            y_train = self.data._y_train
            print("X_train shape in f&p ",X_train.shape)  
            # added DRR
            X_train_fn = self.data._X_train_fn
            
            X_test = X_train
            y_test = y_train
            # added DRR
            X_test_fn = self.data._X_train_fn


            bkys = list(range(3550, 3560))
            wkys = list(range(150, 160))

        elif(self.dataset == "mnist"):
            X_train,y_train = self.get_oneClass_trainData()
            #testing data
            X_test,y_test = self.get_oneClass_trainData()
             
        elif (RCAE_AD.DATASET == "gtsrb"):

            X_train = np.concatenate((self.data._X_train, self.data._X_test))
            y_train = np.concatenate((self.data._y_train, self.data._y_test))

            # X_test = self.data._X_test
            # y_test = self.data._y_test

            trainXPos = X_train[np.where(y_train == 0)]
            trainYPos = np.zeros(len(trainXPos))
            trainXNeg = X_train[np.where(y_train == 1)]
            trainYNeg = 1 * np.ones(len(trainXNeg))

            PosBoundary = len(trainXPos)
            NegBoundary = len(trainXNeg)

            # print("[INFO:]  Length of Positive data", len(trainXPos))
            # print("[INFO:]  Length of Negative data", len(trainXNeg))

            X_train = np.concatenate((trainXPos, trainXNeg))
            y_train = np.concatenate((trainYPos, trainYNeg))
            
            X_test = X_train
            y_test = y_train
            
            # Make sure the axis dimensions are aligned for training convolutional autoencoders
            X_train = np.moveaxis(X_train, 1, 3)
            X_test = np.moveaxis(X_test, 1, 3)
           

            X_train = X_train / 255.0
            X_test = X_test / 255.0

            print("[INFO:] X_train.shape", X_train.shape)
            print("[INFO:] y_train.shape", y_train.shape)

            print("[INFO:] X_test.shape", X_test.shape)
            print("[INFO:] y_test.shape", y_test.shape)
        
        elif((self.dataset == "dogs") or (self.dataset == "catsdogs")or (self.dataset == "adi")):
            print("adi/dogs is the self.dataset")
            X_train = self.data._X_train
            y_train = self.data._y_train
            # added DRR
            X_train_fn = self.data._X_train_fn
            
            # Chalapatay's definition
            X_test = X_train
            y_test = y_train
            # added DRR
            X_test_fn = self.data._X_train_fn
            
            # DRR
            X_tsst = self.data._X_tsst
            y_tsst = self.data._y_tsst
            # added DRR
            X_tsst_fn = self.data._X_tsst_fn

            X_mytest = self.data._X_mytest
            y_mytest = self.data._y_mytest
            X_mytest_fn = self.data._X_mytest_fn
            
        # define lamda set
        lamda_set = [0.0, 0.5, 1.0,  100.0]
        # lamda_set = [0.0,  0.5]
        
        # lamda_set = [0.0]
        # lamda_set = [0.5]
        # mue = 0.0
        # TRIALS = 2
        TRIALS = 4
        ap = np.zeros((TRIALS,))
        auc = np.zeros((TRIALS,))
        myauc = np.zeros((TRIALS,))
        tssauc = np.zeros((TRIALS,))
        prec = np.zeros((TRIALS,))

        for l in range(0, len(lamda_set)):
            # Learn the N using softthresholding technique
            N = 0
            lamda = lamda_set[l]
            print("lamda (Before self.nn_model.compute_softhreshold)=",lamda )
            XTrue = X_train
            print("shape XTrue ", XTrue.shape)
            YTrue = y_train
            print("shape Ytrue ", YTrue.shape)
            XTruefn = X_train_fn 

            # Capture the structural Noise
            print("now  self.nn_model.compute_softhreshold")
            self.nn_model.compute_softhreshold(XTrue, N, lamda, XTrue)
            N = self.nn_model.Noise
             
            print("After self.nn_model.compute_softhreshold")
            print("shape XTrue ", XTrue.shape)
            print("shape N ", N.shape)

            print("Before self.nn_model.cae.predict")
            decoded = self.nn_model.cae.predict(XTrue)
            # rank the best and worst reconstructed images
            [best_top10_keys, worst_top10_keys] = self.nn_model.compute_best_worst_rank(XTrue, decoded)

            print("After self.nn_model.compute_best_worst_rank")
            print("shape XTrue ", XTrue.shape)
            print("shape decoded ", decoded.shape)
            print("\nbest_top10_keys=", best_top10_keys)
            print("\nworst_top10_keys=", worst_top10_keys)
            #DRR
            self.nn_model.visualise_anomalies_detected(XTrue, XTrue, decoded, N, XTruefn, best_top10_keys, worst_top10_keys, lamda)

            # decoded 
            print("shape X_test ", X_test.shape)
            decoded = self.nn_model.cae.predict(X_test)
            print("shape (X_test) decoded ", decoded.shape)

            print("shape X_tsst ", X_tsst.shape)
            tssdecoded = self.nn_model.cae.predict(X_tsst)
            print("shape (X_tsst) tssdecoded ", tssdecoded.shape)

            print("shape X_mytest ", X_mytest.shape)
            mydecoded = self.nn_model.cae.predict(X_mytest)
            print("shape (X_mytest) mydecoded ", decoded.shape)
            XPred = decoded
            
            if(self.dataset == "mnist"):
                decoded = np.reshape(decoded, (len(decoded), 784))
                X_test_for_roc = np.reshape(X_test, (len(X_test), 784))
            
            elif(self.dataset == "cifar10"):
                decoded = np.reshape(decoded, (len(decoded), 3072))
                X_test_for_roc = np.reshape(X_test, (len(X_test), 3072))
                print("shape X_test_for_roc ", X_test_for_roc.shape)
            
            elif(self.dataset == "gtsrb"):
                decoded = np.reshape(decoded, (len(decoded), 3072))
                X_test_for_roc = np.reshape(X_test, (len(X_test), 3072))

            elif((self.dataset == "dogs") or (self.dataset == "catsdogs")or (self.dataset == "adi")):
                decoded = np.reshape(decoded, (len(decoded), 3072))
                X_test_for_roc = np.reshape(X_test, (len(X_test), 3072))
                #
                tssdecoded = np.reshape(tssdecoded, (len(tssdecoded), 3072))
                X_tsst_for_roc = np.reshape(X_tsst, (len(X_tsst), 3072))
                #
                mydecoded = np.reshape(mydecoded, (len(mydecoded), 3072))
                X_mytest_for_roc = np.reshape(X_mytest, (len(X_mytest), 3072))

                
            recErr = ((decoded - X_test_for_roc) ** 2).sum(axis = 1)
            tssrecErr = ((tssdecoded - X_tsst_for_roc) ** 2).sum(axis = 1)
            myrecErr = ((mydecoded - X_mytest_for_roc) ** 2).sum(axis = 1)
            
            print("before save_Most...shape X_test ", X_test.shape)
            #self.save_Most_Anomalous(X_test,recErr)
            #self.save_Most_Normal(X_test,recErr)
            self.save_Most_Anomalous(X_test, X_test_fn, l,0, recErr)
            self.save_Most_Normal(X_test, X_test_fn, l,0, recErr)
            print("before save_Most...shape X_tsst ", X_tsst.shape)
            self.save_Most_Anomalous(X_tsst, X_tsst_fn, l,2, tssrecErr)
            self.save_Most_Normal(X_tsst, X_tsst_fn, l,2, tssrecErr)
            print("before save_Most...shape X_mytest ", X_mytest.shape)
            self.save_Most_Anomalous(X_mytest, X_mytest_fn, l,3, myrecErr)
            self.save_Most_Normal(X_mytest, X_mytest_fn, l,3, myrecErr)
            
            print("shape X_test ", X_test.shape)
            auc[l] = roc_auc_score(y_test, recErr)
            tssauc[l] = roc_auc_score(y_tsst, tssrecErr)
            myauc[l] = roc_auc_score(y_mytest, myrecErr)
            import pandas as pd 
            df = pd.DataFrame(recErr)
            df.to_csv(self.results  +"recErr.csv")
            
            print("=====================")
            print("AUROC", lamda, auc[l])
            print("AUROC-tsst", lamda, tssauc[l])
            print("AUROC-mytest", lamda, myauc[l])
            auc_roc = auc[l]
            print("=======================")
            
            self.save_model(self.nn_model.cae,str(lamda))


        return auc_roc

    def tile_raster_visualise_anomalies_detected(self, testX, testX_fn, worst_top10_keys, TitleStrng): 

        import matplotlib.pyplot as plt
        nrows=10 
        ncols=10
        side = self.IMG_HGT
        channel = self.channel

        worst_top10_keys = list(worst_top10_keys)

        imgpil = np.ndarray(shape=((nrows * ncols),side, side, channel))
        imgNam = ["" for i in range((nrows * ncols))]
        for j in range(0, nrows):
            img = np.ndarray(shape=(side , side , channel))
            for k in range(0, ncols):
               i= nrows * j + k
               img = testX[worst_top10_keys[i]]
               img = 255 * img
               imgpil[i,:,:,:] =  img
               if len(testX_fn[worst_top10_keys[i]] ) > 15:
                    first_chars = testX_fn[worst_top10_keys[i]][0:5]
                    last_chars =  testX_fn[worst_top10_keys[i]][-5:]
                    imgNam[i] = str(i) + str('-') + str(first_chars) + str(last_chars)
               else:
                    imgNam[i] =  str(i) + str('-') + testX_fn[worst_top10_keys[i]]
                   


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
        #fig.tight_layout(pad=1.0)

        # Save the image decoded
        if(self.dataset == "cifar10"):
            # Save the image decoded
            print('Saving Top-100'+ str(TitleStrng) )
            mystrn = str(TitleStrng)  + '_Class'+ str(Cfg.cifar10_normal) + "_Top100.png"
        if(self.dataset == "mnist"):
            # Save the image decoded
            img = np.reshape(img, (side * nrows, side * ncols))
            print('Saving Top-100'+ str(TitleStrng) )
            mystrn =  str(TitleStrng)  + '_Class'+ str(Cfg.mnist_normal) + "_Top100.png"
        if((self.dataset == "dogs") or (self.dataset == "catsdogs")):
            # Save the image decoded
            print('Saving Top-100'+ str(TitleStrng) )
            mystrn =  str(TitleStrng)  + '_Class'+ str(Cfg.dogs_normal) + "_Top100.png"
        if(self.dataset == "adi"):
            # Save the image decoded
            print('Saving Top-100'+ str(TitleStrng) )
            mystrn =  str(TitleStrng)  + '_Class'+ str(Cfg.adi_normal) + "_Top100.png"

        plt.savefig(self.results + str(mystrn), dpi=150)
        plt.clf()
        plt.cla()
        plt.close()


        return 
