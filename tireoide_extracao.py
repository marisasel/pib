import os
import sys
import pydicom
import matplotlib.pyplot as plt
import cv2

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

def visualize_images(images_list):
    for image in images_list:
        plt.imshow(image, cmap = "gray")
        plt.show()        

def main(bmt, graves):

    cropped_bmt_images = []
    cropped_graves_images = []

    adjusted_bmt_images = []
    adjusted_graves_images = []

    filtered_bmt_images = []
    filtered_graves_images = []

    top_hat_bmt_images = []
    top_hat_graves_images = []
    
    binary_bmt_images = []
    binary_graves_images = []

    # LOAD BASES
    bmt_images, bmt_names = load_images(bmt)
    graves_images, graves_names  = load_images(graves)

    # VISUALIZE ORIGINAL IMAGES
    print('BMT dataset: raw images.\n')
    #visualize_images(bmt_images)
    print('Graves dataset: raw images.\n')    
    #visualize_images(graves_images)

    # ROI TESTS
    #for image in bmt_images:
    #    roi = cv2.selectROI('Selecione área de interesse',image,showCrosshair = True,fromCenter = False)
    #    print(roi)
    #    roi_cropped = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    #    cropped_bmt_images.append(roi_cropped)

    #for image in graves_images:
    #    roi = cv2.selectROI('Selecione área de interesse',image,showCrosshair = True,fromCenter = False)
    #    print(roi)
    #    roi_cropped = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    #    cropped_graves_images.append(roi_cropped)

    # VISUALIZE CROPPED IMAGES
    #print('BMT dataset: cropped images.\n')
    #visualize_images(bmt_images)
    #print('Graves dataset: cropped images.\n')    
    #visualize_images(graves_images)

    # FILTER TESTS
    for image in bmt_images:
    #for image in adjusted_bmt_images:
        #aux_img = cv2.blur(image,(3,3))            # apply the average filter via open CV's "blur" function, with 3 x 3 kernel
        #aux_img = cv2.medianBlur(image,3)          # apply the median filter via open CV's "medianBlur" function, with kernel 3
        aux_img = cv2.GaussianBlur(image,(5,5),0)   # apply the gaussian filter via open CV's "GaussianBlur" function, with 5 x 5 kernel
        filtered_bmt_images.append(aux_img)

    for image in graves_images:
    #for image in adjusted_graves_images:
        #aux_img = cv2.blur(image,(3,3))            # apply the average filter via open CV's "blur" function, with 3 x 3 kernel
        #aux_img = cv2.medianBlur(image,3)          # apply the median filter via open CV's "medianBlur" function, with kernel 3
        aux_img = cv2.GaussianBlur(image,(5,5),0)   # apply the gaussian filter via open CV's "GaussianBlur" function, with 5 x 5 kernel
        filtered_graves_images.append(aux_img)

    # VISUALIZE FILTERED IMAGES
    print('BMT dataset: filtered images.\n')
    visualize_images(filtered_bmt_images)
    print('Graves dataset: filtered images.\n')    
    visualize_images(filtered_graves_images)

    # BRIGHTNESS AND CONTRAST TESTS
    alpha = 1.8    # parameter for contrast adjustment
    beta = 8       # parameter for brightness adjustment
    for image in filtered_bmt_images:
        aux_img = cv2.convertScaleAbs(image, alpha = alpha, beta = beta) # adjust the brightness and contrast of the original image
        adjusted_bmt_images.append(aux_img)
 
    for image in filtered_graves_images:
        aux_img = cv2.convertScaleAbs(image, alpha = alpha, beta = beta) # adjust the brightness and contrast of the original image
        adjusted_graves_images.append(aux_img)

    # VISUALIZE ADJUSTED IMAGES
    print('BMT dataset: adjusted images.\n')
    visualize_images(adjusted_bmt_images)
    print('Graves dataset: adjusted images.\n')    
    visualize_images(adjusted_graves_images)

    # TOP-HAT TESTS
    filter_size = (3,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filter_size)
    
    for image in filtered_bmt_images:
    #for image in adjusted_bmt_images:
        aux_img = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,kernel)
        top_hat_bmt_images.append(aux_img)

    for image in filtered_graves_images:
    #for image in adjusted_graves_images:
        aux_img = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,kernel)
        top_hat_graves_images.append(aux_img)

    # VISUALIZE TOP HAT IMAGES
    print('BMT dataset: top hat images.\n')
    visualize_images(top_hat_bmt_images)
    print('Graves dataset: top ha images.\n')    
    visualize_images(top_hat_graves_images)


    # BINARIZATION TESTS
    for image in adjusted_bmt_images:
        ret, aux_img = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
        binary_bmt_images.append(aux_img)

    for image in adjusted_graves_images:
        ret, aux_img = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
        binary_graves_images.append(aux_img)

    # VISUALIZE BINARY IMAGES
    print('BMT dataset: binary images.\n')
    visualize_images(binary_bmt_images)
    print('Graves dataset: binary images.\n')    
    visualize_images(binary_graves_images)


    #print(bmt_names)
    #print(graves_names)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Use: $ python3 tireoide_extracao.py ./BMT ./GRAVES")

	main(sys.argv[1], sys.argv[2])