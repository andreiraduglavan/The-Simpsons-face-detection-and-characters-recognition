import numpy as np
import cv2 as cv
import os

def getYWO(img):
    
    YWO = np.zeros((np.shape(img)[0],
                     np.shape(img)[1],3), dtype=bool)
    YWO[:,:,0] = isYellow(img)
    YWO[:,:,1] = isWhite(img)
    
    # Turns overlappings to zeros (to be classified as 'Other')
    #overlappings = np.greater(np.sum(YWO,axis=2),1)
    #YWO[overlappings,:] = np.zeros(3).astype(bool)
    
    # Sets others as the ones that have all zeros
    others = np.equal(np.sum(YWO,axis=2),0)
    YWO[:,:,2] = others

    
    return YWO

def isYellow(x):

    Blue  = np.greater(100,x[:,:,0])
    Green = np.logical_and(np.greater(x[:,:,1],105), np.greater(255,x[:,:,1]))
    Red   = np.greater(x[:,:,2],105)

    return np.logical_and(np.logical_and(Red,Green), Blue)

def isWhite(x):
    
    Blue  = np.greater(x[:,:,0],180)
    Green = np.greater(x[:,:,1],180)
    Red   = np.greater(x[:,:,2],180)
    
    return np.logical_and(np.logical_and(Red,Green), Blue)

def YWO_to_BGR(YWO):
    
    s0, s1, _ = np.shape(YWO)
    BGR = np.zeros((s0,s1,3), dtype=np.uint8)
    BGR[YWO[:,:,0]] = [0,213,255]
    BGR[YWO[:,:,1]] = [255,255,255]
    BGR[YWO[:,:,2]] = [190,170,100]
    
    return BGR

def show_image(image):
    cv.imshow("",image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    dir_path='../data/exempleValidare/simpsons_validare/'
    i=0
    for file in os.listdir(dir_path):
        filename=dir_path+file
        image=cv.imread(filename)
        imageYOW=getYWO(image)
        show_image(YWO_to_BGR(imageYOW))
        i+=1