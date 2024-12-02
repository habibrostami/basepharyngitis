import os

from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

from settings import IMAGE_WIDTH, IMAGE_HEIGHT
from settings import REF_IMG
REF_IMG = "/mnt/2T/Shojaei/pharyngitis/data/histomatched_images_folders/korea_ref.jpg"
SRC_MAIN_FOLDER = "/mnt/2T/Shojaei/pharyngitis/data/images_folders/"
#SRC_MAIN_FOLDER ="/mnt/2T/Shojaei/pharyngitis/data/korea/"
DEST_FOLDEER = "/mnt/2T/Shojaei/pharyngitis/data/histomatched_images_folders/"

def main():
    ref = cv2.imread(REF_IMG)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)  #  RGB

    for d in os.listdir(SRC_MAIN_FOLDER):
        print(d)
        for im in os.listdir(SRC_MAIN_FOLDER + d):
            src = cv2.imread(SRC_MAIN_FOLDER + d + "/" + im)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # RGB

            matched = exposure.match_histograms(src, ref, channel_axis=-1)
            cv2.imwrite(DEST_FOLDEER+ d + "/" + im, cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_RGB2BGR))

            #print(f'matched shape = {matched.shape}')
            #cv2.imwrite(DEST_FOLDEER+ d + "/" + im,matched)


def average():
    sum_image = None
    count = 0
    for d in os.listdir(SRC_MAIN_FOLDER):

        for im in os.listdir(SRC_MAIN_FOLDER + d):
            image = cv2.imread(SRC_MAIN_FOLDER + d + "/" + im)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #  RGB
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            count += 1
            if sum_image is None:
                sum_image = np.zeros_like(image, dtype=np.float64)

            sum_image += image


    average_image = sum_image / count
    average_image = average_image.astype(np.uint8)
    cv2.imwrite(REF_IMG, cv2.cvtColor(average_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    #average()
    main()