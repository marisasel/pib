import os
import sys
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn 

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def load_dataset(path):
    dataset = []
    for (root, directories, files) in os.walk(path):
        for file in files:
            dcm_path = os.path.join(root, file)
            dicom = pydicom.dcmread(dcm_path)
            dataset.append(dicom.pixel_array)
    return np.array(dataset)                  

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
        if y[i] == 0:
            label = 'BMT'
        else:
            label = 'Graves'
        if i in wrong:
            title = label +" - Wrong"
        else:
            title = label + " - Right"
        plt.title(title)    
    plt.show()

def calculate_scores(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    print(classification_report(y_test, y_pred, labels=[0,1]))
    #generates graphical visualization of the confusion matrix
    #cm = pd.DataFrame(cm, range(2), range(2))
    #plt.figure(figsize = (5.5,4))  
    #sn.set(font_scale = 1.0)
    #sn.heatmap(cm, annot = True, annot_kws = {"size": 12}, fmt = "d") 
    #plt.show() 

def main(dataset_path, features_file):
    
    # Load the Dicom images for visualization
    print("Loading images...")
    bmt = load_dataset(os.path.join(dataset_path, "BMT"))
    graves = load_dataset(os.path.join(dataset_path, "GRAVES"))
    images = np.concatenate((bmt, graves), axis = 0)

    # Load data
    print("Loading data features...")
    X, y = load_svmlight_file(features_file)
    X = X.toarray()
    y = y.astype(int)

    # Select model
    model = GradientBoostingClassifier(n_estimators=10, learning_rate=1, max_depth=1, random_state=0)
      
    # Train and predict using Leave One Out
    print("Training and predicting using Leave One Out models...")
    loo = LeaveOneOut()
    scores = 0
    wrong_predicted = []
    y_pred = []
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
        y_pred.append(model.predict(X_test))
        if model.predict(X_test) == y_test:
            scores += 1
        else:
            wrong_predicted.append(test_index)
    
    print("Scores:", scores, "- Accuracy:", scores/12)
    calculate_scores(y, y_pred)
    print_images(images, y, wrong_predicted)
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: $ python3 thyroid_model.py dataset_path/ <pre_extracted_features_file>")

    main(sys.argv[1], sys.argv[2])