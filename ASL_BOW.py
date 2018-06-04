# -*- coding: utf-8 -*-
"""
Created on Wed May  9 19:54:11 2018

@author: HUSEIN MOHAMUD
"""

import cv2
import numpy as np
import os
from sklearn import svm, grid_search
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.neighbors import KNeighborsClassifier

#%%
# Get the training classes names and store them in a list
train_path = '-----------------------------------------/train'  # put training data location here
training_names = os.listdir(train_path)

def imlist(path):
    """
    The function imlist returns all the names of the files in 
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Create feature extraction and keypoint detector objects
#fea_det = cv2.FeatureDetector_create("SIFT")
surf = cv2.xfeatures2d.SURF_create()

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = surf.detect(im, None)
    kpts, des = surf.compute(im, kpts)
    des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
train_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        train_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (train_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(train_features)
train_features = stdSlr.transform(train_features)
#%% Train dataset----you can either use KNN or SVM by uncommenting the one you want........

param_grid = {'C': [100, 1000, 10000, 100000.0], 'gamma' : [0.001, 0.01, 0.1, 1]}
clf = grid_search.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv= 5)
clf.fit(train_features, np.array(image_classes))
print('The best parameters found by gridSearch: ', clf.best_params_)

#param_grid = {'n_neighbors': list(range(1, 31)), 'weights': ['uniform', 'distance']}
#clf = grid_search.GridSearchCV(KNeighborsClassifier(), param_grid, cv= 10)
#clf.fit(train_features, np.array(image_classes))
#print('The best parameters found by gridSearch: ', clf.best_params_)

result = clf.score(train_features, np.array(image_classes))
print('Training Accuracy: ',result)

# Save the SVM
# save classifier, class names, scaler, number of clusters and vocabulary
joblib.dump((clf, training_names, stdSlr, k, voc), open("trained_data_svm.sav",'wb'))  


#%%
# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("trained_data_svm.sav")

# Get the path of the testing image(s) and store them in a list
image_paths_test = []
image_classes_test = []
class_id = 0

test_path = '-----------------------------------------------------/test'  # put test data location here
testing_names = os.listdir(test_path)

for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imlist(dir)
    image_classes_test+=[class_id]*len(class_path)
    image_paths_test+=class_path
    class_id+=1
    
# Create feature extraction and keypoint detector objects
surf = cv2.xfeatures2d.SURF_create()

# List where all the descriptors are stored
des_list_test = []

for image_path in image_paths_test:
    im = cv2.imread(image_path)
    kpts = surf.detect(im, None)
    kpts, des = surf.compute(im, kpts)
    des_list_test.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list_test[0][1]
for image_path, descriptor in des_list_test[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

# 
test_features = np.zeros((len(image_paths_test), k), "float32")
for i in range(len(image_paths_test)):
    words, distance = vq(des_list_test[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths_test)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_features = stdSlr.transform(test_features)

result = clf.score(test_features, np.array(image_classes_test))
print('Test Accuracy: ',result)

#%% plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
predictions = clf.fit(train_features, np.array(image_classes)).predict(test_features)
cnf_matrix = confusion_matrix(image_classes_test, predictions)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=classes_names, normalize=True,
                      title='Normalized confusion matrix')

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=classes_names,
                      title='Confusion matrix, without normalization')
plt.show()

#%% predicting with images
# Perform the predictions
predic =  [classes_names[i] for i in clf.predict(test_features)]
# Visualize the results
for image_path, prediction in zip(image_paths_test, predic):
    image = cv2.imread(image_path)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    pt = (0, 3 * image.shape[0] // 4)
    cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_TRIPLEX, 8, [255, 0, 0], 4)
    cv2.imshow("Image", image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()