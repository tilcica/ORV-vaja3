import cv2 as cv
import numpy as np
import random

def kmeans(image, k=3, iterations=10, choice='r', T=10):
    pass

def meanshift(image, velikost_okna=30, dimenzija=3):
    pass

def izracunaj_centre(image, choice, centerDimension, T):
    pass


if __name__ == "__main__":
    image = cv.imread('image.png')
    #segmented = meanshift(image, velikost_okna=30, dimenzija=5)
    segmented = kmeans(image)
    cv.imshow('Segmented', segmented)
    cv.waitKey(0)