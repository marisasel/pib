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
            #img_info = pydicom.read_file(file_path)
            #print(img_info)
            img = pydicom.dcmread(file_path)
            arr = img.pixel_array
            plt.imshow(arr, cmap="gray")
            plt.show()
            if img is not None:
                images.append(img) 
                names.append(file)
    return images, names

def main(bmt, graves):

    bmt_images, bmt_names = load_images(bmt)
    graves_images, graves_names  = load_images(graves)

    print(bmt_images, bmt_names)
    print(graves_images, graves_names)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Use: $ python3 tireoide.py ./BMT ./GRAVES")

	main(sys.argv[1], sys.argv[2])