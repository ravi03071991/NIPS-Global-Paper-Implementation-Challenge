import keras
import numpy as np
import pandas as pd
from keras.applications import vgg16, resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import math
import scipy
from scipy.stats import binom
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')

#Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')

# Load the pickle file for obtaining the classes
DAT_FILE_PATH = "imagenet_val/imagenet-classes-dict.dat" 
pickle_in = open(DAT_FILE_PATH, "rb")
classes_dict = pickle.load(pickle_in)

# Load validation set ground truth labels
LABELS_FILE_PATH = 'imagenet_val/ILSVRC2012_validation_ground_truth.txt'
y = np.loadtxt(LABELS_FILE_PATH, skiprows=0)

# Get .JPEG file names
DIR_PATH = "imagenet_val/val/"
filelist = os.listdir(DIR_PATH)
filelist = sorted(filelist)

# Predict the probabilities and labels for validation samples
predict_prob = []
predict_label = []

for i in range(len(y)):

    filename = DIR_PATH + filelist[i]
    # load an image in PIL format
    original = load_img(filename)
    #print('PIL image size',original.size)
    #plt.imshow(original)
    #plt.show()
    #wpercent = (basewidth/float(original.size[0]))
    #hsize = int((float(original.size[1])*float(wpercent)))
    #original = original.resize((basewidth,hsize), Image.ANTIALIAS)
    
    aspect_ratio = original.size[0]/original.size[1]
    
    if original.size[0] < original.size[1]:
        width = 256
        height = width/aspect_ratio
    else:
        height = 256
        width = height * aspect_ratio
    
    original = original.resize((int(width), int(height)))
        
    
    width, height = original.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    original = original.crop((left, top, right, bottom))

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    

    # prepare the image for the VGG model
    processed_image = preprocess_input(image_batch)
 
    # get the predicted probabilities for each class
    predictions = resnet_model.predict(processed_image)
    label = decode_predictions(predictions)
    predict_prob.append(np.max(predictions))
    predict_label.append(classes_dict[label[0][0][0]])
    #print(classes_dict[label[0][0][0]], y[i])

# Convert predict_prob, predict_label to numpy arrays
predict_prob = np.array(predict_prob)
predict_label = np.array(predict_label)

# 0 for correct prediction and 1 for wrong prediction
residuals = (predict_label!=y)

# Check the accuracy
Accuracy = ((50000 - sum(residuals))/50000)*100

print("Accuracy is: ", Accuracy)

# Splitting the validation dataset for training and testing SGR algorithm
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=8)

for train_index, test_index in sss.split(predict_prob, y):
    prob_train, prob_test = predict_prob[train_index], predict_prob[test_index]
    residuals_train, residuals_test = residuals[train_index], residuals[test_index]

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

# Define confidence level
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
    desiredrisk, trainrisk, traincov, testrisk, testcov, riskbound = SGR(rstar[i],delta, prob_train, prob_test, residuals_train, residuals_test)

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

