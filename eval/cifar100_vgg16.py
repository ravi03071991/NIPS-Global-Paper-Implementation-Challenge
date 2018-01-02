import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(8)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(80)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(800)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import pandas as pd
import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten, Dense, Activation
from keras import optimizers
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import math
from scipy.stats import binom
import scipy

class VGG16_CIFAR100:
    
    def __init__(self):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.batch_size = 128
        self.epoches = 250
        self.learning_rate = 0.1
        self.lr_decay = 1e-6
    
    # Function to create dataset for training and validation of model
    def create_dataset(self): 
        
        num_classes = self.num_classes

        # Create Train and Test datasets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        # Normalize the data
        x_train, x_test = self.normalize(x_train, x_test)
        
        # Create one-hot encodings
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        
        return x_train, y_train, x_test, y_test
    
    # Function to normalize train and validation datasets
    def normalize(self,X_train,X_test): 
        
        # Compute Mean
        mean = np.mean(X_train,axis=(0, 1, 2, 3))
        
        # Compute Standard Deviation
        std = np.std(X_train, axis=(0, 1, 2, 3)) 
        
        # Normalize the data
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        
        return X_train, X_test
        
    # Function to build the model   
    def buildmodel(self): 
        
        weight_decay = self.weight_decay
        num_classes = self.num_classes
        x_shape = self.x_shape
        
        model = Sequential()
    
        # First group of convolutional layer
        
        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape = x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Second group of convolutional layer

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Third group of convolutional layer

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Fourth group of convolutional layer

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Fifth group of convolutional layer

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Two Fully connected layer

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        
        return model
    
    # Function to train the model
    def model_train(self, model, x_train, y_train, x_test, y_test, weights):
        
        if weights: # If model weights are already avaialble
            model.load_weights('cifar100_vgg16.h5')
        else:

            # Training parameters
            batch_size = self.batch_size
            number_epoches = self.epoches
            learning_rate = self.learning_rate
            lr_decay = self.lr_decay

            # Data augmentation
            dataaugmentation = ImageDataGenerator(
                                    featurewise_center=False,  # set input mean to 0 over the dataset
                                    samplewise_center=False,  # set each sample mean to 0
                                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    samplewise_std_normalization=False,  # divide each input by its std
                                    zca_whitening=False,  # apply ZCA whitening
                                    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                    horizontal_flip=True,  # randomly flip images
                                    vertical_flip=False)  # randomly flip images
        
            dataaugmentation.fit(x_train)

            # Optimization details
            sgd = optimizers.SGD(lr=0.0, decay=lr_decay, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


            # Function to reduce learning rate by half after every 25 epochs
            def step_decay(epoch):
        
                # LearningRate = InitialLearningRate * DropRate^floor(Epoch / EpochDrop)
        
                initial_lrate = 0.1
                drop = 0.5
                epochs_drop = 25.0
                lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
                return lrate

            # Callback for learning rate schedule
            lrate = LearningRateScheduler(step_decay)
            callbacks_list = [lrate]

            # spe = Steps per epoch
            spe = x_train.shape[0] // batch_size
        
            # Fit the model
            model.fit_generator(dataaugmentation.flow(x_train, y_train,
                                                 batch_size = batch_size),
                                    steps_per_epoch = spe, callbacks=callbacks_list,
                                    epochs = number_epoches,
                                    validation_data = (x_test, y_test))
        
            # Save model weights
            model.save_weights('cifar100_vgg16.h5')
        
        return model

# Create class object
model_cifar100 = VGG16_CIFAR100()

# Training and validation datasets
x_train, y_train, x_test, y_test = model_cifar100.create_dataset()

# Create model
model = model_cifar100.buildmodel()

# Train the model
model = model_cifar100.model_train(model, x_train, y_train, x_test, y_test, weights = True)

# Prediction on test set
predict_test = model.predict(x_test)

# Get highest probability on test set
predict_test_prob = np.max(predict_test,1)

# 0 for correct prediction and 1 for wrong prediction
residuals = (np.argmax(predict_test,1) != np.argmax(y_test,1))

# Loss computation
loss = (-1)*((residuals*np.log10(predict_test_prob)) + ((1-residuals)*np.log(1-predict_test_prob)))

# Checking validation accuracy is matching with our calculations
Accuracy = ((10000 - sum(residuals))/10000)*100

print("Accuracy is: ", Accuracy)

# Splitting the validation dataset for training and testing SGR algorithm
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=8)

for train_index, test_index in sss.split(x_test, y_test):
    sgr_x_train, sgr_x_test = x_test[train_index], x_test[test_index]
    sgr_y_train, sgr_y_test = y_test[train_index], y_test[test_index]

# Prediction on SGR train set
predict_sgr_train = model.predict(sgr_x_train)

# Get highest probability on SGR train set
predict_sgr_train_prob = np.max(predict_sgr_train,1)

# 0 for wrong prediction and 1 for correct prediction for SGR train set
residuals_sgr_train = (np.argmax(predict_sgr_train,1)!=np.argmax(sgr_y_train,1))

# Loss computation on SGR train set
loss_sgr_train = (-1)*((residuals_sgr_train*np.log10(predict_sgr_train_prob)) + ((1-residuals_sgr_train)*np.log(1-predict_sgr_train_prob)))

# Prediction on SGR test set
predict_sgr_test = model.predict(sgr_x_test)

# Get highest probability on SGR test set
predict_sgr_test_prob = np.max(predict_sgr_test,1)

# 0 for wrong prediction and 1 for correct prediction for SGR test set
residuals_sgr_test = (np.argmax(predict_sgr_test,1)!=np.argmax(sgr_y_test,1))

# Loss computation on SGR test set
loss_sgr_test = (-1)*((residuals_sgr_test*np.log10(predict_sgr_test_prob)) + ((1-residuals_sgr_test)*np.log(1-predict_sgr_test_prob)))

def calculate_bound(delta, m, risk):
        
        epsilon = 1e-7
        
        x = risk         # Lower bound
        z = 1            # Upper bound
        y = (x + z)/2    # mid point
        
        epsilonhat  = (-1*delta) + scipy.stats.binom.cdf(int(m*risk), m, y)
        
        while abs(epsilonhat)>epsilon:
            if epsilonhat>0:
                x = y
            else:
                z = y
                
            y = (x + z)/2
            #print("x", x)
            #print("y", y)
            epsilonhat  = (-1*delta) + scipy.stats.binom.cdf(int(m*risk), m, y)
            #print(epsilonhat)
        return y

def SGR(targetrisk, delta, predict_sgr_train_prob, predict_sgr_test_prob, residuals_sgr_train, residuals_sgr_test):

        # Number of training samples for SGR algorithm
        m = len(residuals_sgr_train)

        # Sort the probabilities
        probs_idx_sorted = np.argsort(predict_sgr_train_prob)

        zmin = 0
        zmax = m-1
        deltahat = delta/math.ceil(math.log2(m))

        for i in range(math.ceil(math.log2(m) + 1)):
            
            #print("iteration", i)

            mid = math.ceil((zmin+zmax)/2)

            mi = len(residuals_sgr_train[probs_idx_sorted[mid:]])
            theta = predict_sgr_train_prob[probs_idx_sorted[mid]]
            trainrisk = sum(residuals_sgr_train[probs_idx_sorted[mid:]])/mi
            
            
            testrisk = (sum(residuals_sgr_test[predict_sgr_test_prob>=theta]))/(len(residuals_sgr_test[predict_sgr_test_prob>=theta])+1)
            testcoverage = (len(residuals_sgr_test[predict_sgr_test_prob>=theta]))/(len(predict_sgr_test_prob))
                
            
            bound = calculate_bound(deltahat, mi, trainrisk)
            traincoverage = mi/m
            
            if bound>targetrisk:
                zmin = mid
            else:
                zmax = mid

        return targetrisk, trainrisk, traincoverage, testrisk, testcoverage, bound

# Define confidence level parameter delta
delta = 0.001

desired_risk = []
train_risk = []
train_coverage = []
test_risk = []
test_coverage = []
risk_bound = []

# Different desired risk values
rstar = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]

# Testing the SGR algorithm for different desired risk values

for i in range(len(rstar)):
    
    # For desired risk 0.01
    desiredrisk, trainrisk, traincov, testrisk, testcov, riskbound = SGR(rstar[i],delta, predict_sgr_train_prob, predict_sgr_test_prob, residuals_sgr_train, residuals_sgr_test)

    # Append the values to the list
    desired_risk.append(desiredrisk)
    train_risk.append(trainrisk)
    train_coverage.append(traincov)
    test_risk.append(testrisk)
    test_coverage.append(testcov)
    risk_bound.append(riskbound)

Result = [('Desired Risk', desired_risk) ,
          ('Train Risk', train_risk),
          ('Train Coverage', train_coverage),
          ('Test Risk', test_risk),
          ('Test Coverage', test_coverage),
          ('Risk bound', risk_bound)]

Result = pd.DataFrame.from_items(Result)
print(Result)

