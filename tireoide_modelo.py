import os
import sys
import pydicom
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA


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
    dataset_numpy = []

    for sample in dataset:
        dataset_numpy.append(sample.pixel_array)

    return np.array(dataset_numpy)



def main(dataset_path):
    # Load the dicom images
    bmt = load_dataset(os.path.join(dataset_path, "BMT"))
    graves = load_dataset(os.path.join(dataset_path, "GRAVES"))

    # Concatenate dataset
    X = bmt + graves

    # Preprocess data
    X = preprocess(X)
    # Aplicar processamentos aqui, antes de dar FLATTEN e RESHAPE
    X = X.flatten().reshape(X.shape[0], X.shape[1]*X.shape[2])

    # Create labels 
    y = []
    for i in range(0, len(bmt)):
        y.append('bmt')
    for i in range(0, len(graves)):
        y.append('graves')
    y = np.array(y)

    # Select Model
    model = svm.SVC(kernel='rbf')

    # Apply PCA for dimensionality reduction
    pca = PCA()
    pca.fit(X)
    X = pca.transform(X)
   
    
    # Train and predict using Leave One Out
    loo = LeaveOneOut()
    acertos = 0
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
        print ("predito:", model.predict(X_test))
        print ("era:", y_test)
        print("--")

        if model.predict(X_test) == y_test:
            acertos += 1
    
    print("acertos:", acertos, "-", acertos/12*100, "%")
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: $ python3 tireoide.py dataset_path/")

    main(sys.argv[1])