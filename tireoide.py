import os
import sys
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.datasets import load_svmlight_file

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier


def load_dataset(path):
    dataset = []
	
    for (root, directories, files) in os.walk(path):
        for file in files:
            dcm_path = os.path.join(root, file)
            dataset.append(pydicom.dcmread(dcm_path))          

    return dataset

# não finalizado
# def plot_dataset(dataset):
#     for image in dataset:
#         plt.imshow(dataset[0].pixel_array)
#         plt.show()

def preprocess(dataset):
# Applying filters and extracting pixels 

    dataset_numpy = []

    for sample in dataset:
        img = sample.pixel_array

        # Blur to remove noise
        img_filtered = cv2.blur(img,(3,3)) 

        # Increase constrast
        img_filtered = cv2.convertScaleAbs(img_filtered, alpha=1.5, beta=0)

        # Erosion to better segmentation
        img_filtered = cv2.erode(img_filtered, (3,3))

        dataset_numpy.append(img_filtered)

    return np.array(dataset_numpy)

def print_images(images, y, wrong):
    # create figure
    fig = plt.figure()
    
    # setting values to rows and column variables
    rows = 3
    columns = 4

    for i in range(0, images.shape[0]):
        fig.add_subplot(rows, columns, i+1)
        # showing image
        plt.imshow(images[i])
        plt.axis('off')

        if i in wrong:
            title = y[i] +" - Wrong"
        else:
            title = y[i] + " - Right"

        plt.title(title)
    
    plt.show()


#def main(dataset_path):
def main(data):

    # Load the dicom images
    #bmt = load_dataset(os.path.join(dataset_path, "BMT"))
    #graves = load_dataset(os.path.join(dataset_path, "GRAVES"))

    # Concatenate dataset
    #X = bmt + graves

    # Preprocess data
    #X = preprocess(X)
    #images = X.copy()
    #X = X.flatten().reshape(X.shape[0], X.shape[1]*X.shape[2])

    # Create labels 
    #y = []
    #for i in range(0, len(bmt)):
    #    y.append('bmt')
    #for i in range(0, len(graves)):
    #    y.append('graves')
    #y = np.array(y)
    
    # Loads data
    print ("Loading data...")
    X, y = load_svmlight_file(data)

    # Select Model
    model = GradientBoostingClassifier(n_estimators=10, learning_rate=1, max_depth=1, random_state=0)

    # Apply PCA for dimensionality reduction
    #pca = PCA()
    #pca.fit(X)
    #X = pca.transform(X)
   
    
    # Train and predict using Leave One Out
    loo = LeaveOneOut()
    acertos = 0
    wrong_predicted = []
    for train_index, test_index in loo.split(X):

        X_test = X[test_index]
        y_test = y[test_index]

        X_train = []
        y_train = []

        for index in train_index:
            X_train.append(X[index])
            y_train.append(y[index])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        model.fit(X_train, y_train)

        if model.predict(X_test) == y_test:
            acertos += 1
        else:
            wrong_predicted.append(test_index)
    
    print("acertos:", acertos, "-", acertos/12*100, "%")
    #print_images(images, y, wrong_predicted)
        

#if __name__ == "__main__":
#    if len(sys.argv) != 2:
#        sys.exit("Use: $ python3 tireoide.py dataset_path/")

#    main(sys.argv[1])

if __name__ == "__main__":
        if len(sys.argv) != 2:
                sys.exit("Use: $ python3 tireoide.py <data>")

        main(sys.argv[1])