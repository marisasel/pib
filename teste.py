import pydicom
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


ds = pydicom.dcmread('image_itk_mediana_5.dcm')
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone)  # set the color map to bone
print(ds)
img = ds.pixel_array
#print(img.shape)
print(img)
print(img.min())
print(img.max())
plt.imshow(img, cmap = plt.cm.gray)
plt.show()
plt.hist(img)
plt.show()

dicom = pydicom.read_file('image_itk_mediana_3.dcm')
rows = dicom.get(0x00280010).value
cols = dicom.get(0x00280011).value
#print(img.min())
#print(img.max())

#img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR565)
#cv.imshow('RGB', img_bgr)
#cv.waitKey()

#new_img = np.zeros((rows, cols), np.uint8)
#for i in range(0, rows):
#    for j in range(0, cols):
#        pixel_val = img[i][j]
        #checar se tem de reduzir dimensões de cores
#        new_img[i][j] = pixel_val

#cv.imwrite("nova_image.tiff", new_img)
#cv.imshow('new_img', new_img)
#print(new_img.min())
#print(new_img.max())
#plt.hist(new_img)
#plt.show()

#img3 = np.zeros((rows, cols), np.uint8)
#img3 = cv.equalizeHist(new_img)
#cv.imshow('new_img equalizada', img3)
#plt.hist(img3)
#plt.show()


#img2 = cv.equalizeHist(img)
#cv.imshow(img2)
#plt.hist(img2)
#plt.show()