import os
import sys
import pydicom
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.decomposition import PCA

def load_images(directory):
    images = []
    names = []   
    for (root, directories, files) in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(root)+'/'+file
            dataset = pydicom.dcmread(file_path)
            img = dataset.pixel_array
            if img is not None:
                images.append(img) 
                names.append(file)
    return images, names

def visualize_images(img_list):
    for image in img_list:
        plt.imshow(image, cmap = "gray")
        plt.show()

def pre_process(dataset):
# Applying filters and extracting pixels 
    dataset_numpy = []
    for sample in dataset:
        #img = sample.pixel_array
        # Blur to remove noise
        img_filtered = cv2.blur(sample,(3,3)) 
        # Increase constrast
        img_filtered = cv2.convertScaleAbs(img_filtered, alpha=1.5, beta=0)
        # Erosion to better segmentation
        img_filtered = cv2.erode(img_filtered, (3,3))
        dataset_numpy.append(img_filtered)
    return np.array(dataset_numpy)

def crop_bmt(img_list):
    cropped_bmt = []
    cropped_bmt.append(img_list[0][50:87, 46:85])
    cropped_bmt.append(img_list[1][58:94, 40:83])
    cropped_bmt.append(img_list[2][48:82, 46:82])
    cropped_bmt.append(img_list[3][38:63, 53:83])
    cropped_bmt.append(img_list[4][60:89, 48:85])
    cropped_bmt.append(img_list[5][46:84, 46:84])
    return cropped_bmt

def crop_graves(img_list):
    cropped_graves = []
    cropped_graves.append(img_list[0][41:76, 49:82])
    cropped_graves.append(img_list[1][43:78, 48:83])
    cropped_graves.append(img_list[2][39:73, 42:75])
    cropped_graves.append(img_list[3][43:77, 47:83])
    cropped_graves.append(img_list[4][48:83, 43:79])
    cropped_graves.append(img_list[5][49:85, 47:78])
    return cropped_graves

def remove_noise(img_list):
    filtered_images = []
    for image in img_list:
        #aux_img = cv2.blur(image,(3,3))            # apply the average filter via open CV's "blur" function, with 3 x 3 kernel
        #aux_img = cv2.medianBlur(image,3)          # apply the median filter via open CV's "medianBlur" function, with kernel 3
        aux_img = cv2.GaussianBlur(image,(5,5),0)   # apply the gaussian filter via open CV's "GaussianBlur" function, with 5 x 5 kernel
        filtered_images.append(aux_img)
    return filtered_images

def adjust_brightness_contrast(img_list):
    adjusted_images = []
    alpha = 1.8                                     # parameter for contrast adjustment
    beta = 8                                        # parameter for brightness adjustment
    for image in img_list:
        aux_img = cv2.convertScaleAbs(image, alpha = alpha, beta = beta) # adjust the brightness and contrast of the original image
        adjusted_images.append(aux_img)
    return adjusted_images

def top_hat(img_list):
    top_hat_images = []
    filter_size = (3,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filter_size)    
    for image in img_list:
        aux_img = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,kernel)
        top_hat_images.append(aux_img)
    return top_hat_images

def binarize_otsu(img_list):
    binary_images = []
    for image in img_list:
        ret, aux_img = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
        binary_images.append(aux_img)
    return binary_images

def hu_moments_descriptors(img_list):
    hu_descriptors = []
    for image in img_list:
        hu_desc_array = cv2.HuMoments(cv2.moments(image)).flatten()
        hu_descriptors.append(hu_desc_array)
    return hu_descriptors

def lbp_descriptors(img_list):
    radius = 3 
    n_points = 8 * radius
    n_bins = np.arange(0, n_points + 3)
    range = (0, n_points + 2)
    eps = 1e-7
    METHOD = 'uniform'
    lbp_images = []
    lbp_hist = []
    for image in img_list:
        # to build the histogram of patterns
        lbp = local_binary_pattern(image, n_points, radius, METHOD)
        (hist, _) = np.histogram(lbp.ravel(), bins = n_bins, range = range)
        # to normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        lbp_images.append(lbp)
        lbp_hist.append(hist)
    return lbp_images, lbp_hist

def histogram_descriptors(img_list):
    hist_images = []
    for image in img_list:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_images.append(hist)
    return hist_images 

def min_max_normalization(img_list):
    norm_images = []
    for image in img_list:
        max = np.max(image)
        min = np.min(image)
        norm = (image - min) / (max - min)
        norm_images.append(norm)
    return norm_images

def central_trends_features(img_list):
    central_trends = []
    for image in img_list:
        features = []
        features.append(np.mean(image))
        features.append(np.std(image))
        features.append(np.median(image))
        central_trends.append(features)
    return central_trends

def erosion(img_list):
    eroded = []
    for image in img_list:
        inverted = cv2.bitwise_not(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        ero_img = cv2.erode(image, kernel , iterations = 2)
        eroded.append(ero_img)
    return eroded

def count_objetcs(img_list):
    num_obj = []
    for image in img_list:
        edge = cv2.Canny(image, 70, 150)          # edge detection with Canny
        obj, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num = len(obj)
        num_obj.append(num)   
    return num_obj

def concatenate_features(complete, hu, top_hat_hu, lbp, hist_binary, ct, num_obj, num_obj_eroded):
    features = []
    dataset_size = len(hu)
    i = 0
    while i < dataset_size:
        img_features = []
        for element in complete[i]:
            img_features.append(element)        
        for element in hu[i]:
            img_features.append(element)
        for element in top_hat_hu[i]:
            img_features.append(element)    
        for element in lbp[i]:
            img_features.append(element)
        img_features.append(int(hist_binary[i][255]))
        for element in ct[i]:
            img_features.append(element) 
        img_features.append(num_obj[i])
        img_features.append(num_obj_eroded[i])
        features.append(img_features)
        i += 1
    return features

def generate_svm_file(bmt_features, graves_features, file):    
    # writting bmt dataset features
    dataset_bmt_size = len(bmt_features)
    i = 0
    while i < dataset_bmt_size:
        file.write(str("0") + " ")
        k = 0
        for feature in bmt_features[i]:
            file.write(str(k) + ":" + str(feature) + " ")
            k += 1
        file.write("\n")
        i += 1
    # writting graves dataset features
    dataset_graves_size = len(graves_features)
    j = 0
    while j < dataset_graves_size:
        file.write(str("1") + " ")
        k = 0
        for feature in graves_features[j]:
            file.write(str(k) + ":" + str(feature) + " ")
            k += 1
        file.write("\n")
        j += 1        
    return

def main(bmt, graves):
    
    # LOAD BASES
    print ('Loading images...')
    bmt_images, bmt_names = load_images(bmt)
    graves_images, graves_names  = load_images(graves)

    # PRE-PROCESS RAW IMAGES
    bmt_complete = pre_process(bmt_images)
    graves_complete = pre_process(graves_images)
    bmt_complete = bmt_complete.flatten().reshape(bmt_complete.shape[0], bmt_complete.shape[1] * bmt_complete.shape[2])
    graves_complete = graves_complete.flatten().reshape(graves_complete.shape[0], graves_complete.shape[1] * graves_complete.shape[2])

    # APPLY PCA FOR DIMENSIONALITY REDUCTION
    pca = PCA()
    pca.fit(bmt_complete)
    bmt_complete = pca.transform(bmt_complete)
    pca.fit(graves_complete)
    graves_complete = pca.transform(graves_complete)

    # CROP IMAGES
    print ('Cropping images...')
    cropped_bmt_images = crop_bmt(bmt_images)
    cropped_graves_images = crop_graves(graves_images) 

    # PRE-PROCESS IMAGES
    print ('Pre-processing images...')

    # REMOVE NOISE
    filtered_bmt_images = remove_noise(cropped_bmt_images)
    filtered_graves_images = remove_noise(cropped_graves_images)

    # BRIGHTNESS AND CONTRAST ADJUSTMENTS
    adjusted_bmt_images = adjust_brightness_contrast(filtered_bmt_images)
    adjusted_graves_images = adjust_brightness_contrast(filtered_graves_images)

    # TOP-HAT TRANSFORMATION
    top_hat_bmt_images = top_hat(filtered_bmt_images)
    top_hat_graves_images = top_hat(filtered_graves_images)

    # BINARIZATION USING OTSU
    binary_bmt_images = binarize_otsu(adjusted_bmt_images)
    binary_graves_images = binarize_otsu(adjusted_graves_images)
    
    # FEATURES EXTRACTION
    print ('Extracting features...')
    bmt_hu_descriptors = hu_moments_descriptors(cropped_bmt_images)
    graves_hu_descriptors = hu_moments_descriptors(cropped_graves_images)

    top_hat_bmt_hu_descriptors = hu_moments_descriptors(top_hat_bmt_images)
    top_hat_graves_hu_descriptors = hu_moments_descriptors(top_hat_graves_images)

    lbp_bmt, lbp_bmt_descriptors = lbp_descriptors(filtered_bmt_images)
    lbp_graves, lbp_graves_descriptors = lbp_descriptors(filtered_graves_images)

    hist_binary_bmt_images = histogram_descriptors(binary_bmt_images)
    hist_binary_graves_images = histogram_descriptors(binary_graves_images)

    norm_bmt = min_max_normalization(adjusted_bmt_images)
    norm_graves = min_max_normalization(adjusted_graves_images)

    ct_bmt = central_trends_features(norm_bmt)
    ct_graves = central_trends_features(norm_graves)

    eroded_bmt_images = erosion(binary_bmt_images)
    eroded_graves_images = erosion(binary_graves_images)

    num_obj_bmt = count_objetcs(binary_bmt_images)
    num_obj_graves = count_objetcs(binary_graves_images)
    num_obj_bmt_eroded = count_objetcs(eroded_bmt_images)
    num_obj_graves_eroded = count_objetcs(eroded_graves_images)

    bmt_features = concatenate_features(bmt_complete, bmt_hu_descriptors, top_hat_bmt_hu_descriptors, lbp_bmt_descriptors, hist_binary_bmt_images, ct_bmt, num_obj_bmt, num_obj_bmt_eroded)
    graves_features = concatenate_features(graves_complete, graves_hu_descriptors, top_hat_graves_hu_descriptors, lbp_graves_descriptors, hist_binary_graves_images, ct_graves, num_obj_graves, num_obj_graves_eroded)

    fout = open("features.txt","w")
    generate_svm_file(bmt_features, graves_features, fout)
    fout.close
    print ('Done. Take a look into features.txt')

    print('BMT images order:')
    print(bmt_names)
    print('Graves images order:')
    print(graves_names)
    
if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Use: $ python3 thyroid_extraction.py ./BMT ./GRAVES")

	main(sys.argv[1], sys.argv[2])