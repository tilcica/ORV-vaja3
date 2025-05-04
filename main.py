import cv2 as cv
import numpy as np
import random

def kmeans(image, k=3, iterations=10, choice='r', T=10):
    centers = np.float32(izracunaj_centre(image, choice=choice, centerDimension=3, T=T))
    
    h, w, c = image.shape
    pixelValues = np.float32(image.reshape((-1, 3)))
    
    for _ in range(iterations):
        distances = np.sqrt((pixelValues[:, np.newaxis] - centers)**2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        
        newCenters = np.zeros_like(centers)
        for i in range(k):
            clusterPixels = pixelValues[labels == i]
            if len(clusterPixels) > 0:
                newCenters[i] = clusterPixels.mean(axis=0)
        
        if np.allclose(centers, newCenters):
            break
            
        centers = newCenters
    
    segmentedCenters = centers.astype(np.uint8)
    segmentedImage = segmentedCenters[labels].reshape((h, w, c))
    
    return(segmentedImage)

def meanshift(image, velikost_okna=30, dimenzija=3):
    h, w, c = image.shape

    if dimenzija == 3:
        features = image.reshape((-1, 3))
    elif dimenzija == 5:
        Y, X = np.mgrid[0:h, 0:w]
        features = np.hstack(((image.reshape((-1, 3)))/255.0, np.dstack((X/w, Y/h)).reshape((-1, 2))))
    
    features = np.float32(features)
    
    def gaussian_kernel(distanceSqr, heightSqr):
        return np.exp(-distanceSqr / (2 * heightSqr))
    
    segmentedFeatures = features.copy()
    for i in range(len(features)):
        print(i)
        xI = features[i]
        for _ in range(10):
            weights = gaussian_kernel(np.sum((features - xI)**2, axis=1), velikost_okna ** 2)
            
            if np.sum(weights) == 0:
                break
                
            xInew = np.sum(features * weights[:, np.newaxis], axis=0) / np.sum(weights)
            
            if np.linalg.norm(xInew - xI) < 1e-5:
                break
                
            xI = xInew
        
        segmentedFeatures[i] = xI

    return(((segmentedFeatures[:, :3] * 255).clip(0, 255).astype(np.uint8)).reshape((h, w, c)))

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
        maxAttempts = 1000
        attempts = 0
        
        while len(centers) < centerDimension and attempts < maxAttempts:
            candidate = features[random.randint(0, len(features) - 1)]
            
            valid = True
            for center in centers:
                if np.linalg.norm(candidate[:3] - center[:3]) < T:
                    valid = False
                    break
            
            if valid:
                centers.append(candidate)
                attempts = 0
            else:
                attempts += 1
        
        if attempts >= maxAttempts:
            print("took too many attempts (1000+) to find centers")
    
    return np.array(centers)


if __name__ == "__main__":
    image = cv.imread('image.png')
    #segmented = meanshift(image, velikost_okna=30, dimenzija=5)
    segmented = kmeans(image)
    cv.imshow('Segmented', segmented)
    cv.waitKey(0)