import cv2 as cv
import numpy as np
import random

def kmeans(image, k=3, iterations=10, choice='r', T=10):
    pass

def meanshift(image, velikost_okna=30, dimenzija=3):
    pass

def izracunaj_centre(image, choice, centerDimension, T):
    centers = []
    if centerDimension == 3:
        features = image.reshape((-1, 3))
    elif centerDimension == 5:
        h, w = image.shape[:2]
        Y, X = np.mgrid[0:h, 0:w]
        features = np.hstack((image.reshape((-1, 3)), np.dstack((X, Y)).reshape((-1, 2))))
    
    features = np.float32(features)
    
    if choice == 'r':
        displayImg = image.copy()
    
        def selectCenter(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                if centerDimension == 3:
                    center = image[y, x]
                else:
                    center = np.append(image[y, x], [x, y])
                centers.append(center)
                cv.circle(displayImg, (x, y), 5, (0, 0, 255), -1)
                cv.imshow('center select', displayImg)
    
        cv.imshow('center select', displayImg)
        cv.setMouseCallback('center select', selectCenter)
        
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        
        cv.destroyAllWindows()

    elif choice == 'n':
        pass
    
    return np.array(centers)


if __name__ == "__main__":
    image = cv.imread('image.png')
    #segmented = meanshift(image, velikost_okna=30, dimenzija=5)
    segmented = kmeans(image)
    cv.imshow('Segmented', segmented)
    cv.waitKey(0)