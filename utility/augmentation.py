#Name: Aisangam
#Url: http://www.aisangam.com/
#Blog:http://www.aisangam.com/blog/
#Company: Aisangam
#YouTube Channel Link: https://www.youtube.com/channel/UC9x_PL-LPk3Wp5V85F4GLHQ
#Discription: https://youtu.be/PePk_YkMQn0?list=PLCK5Mm9zwPkFt1iX30kD5eJ9hy-EeijQn

import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np

#co-relation between Opencv and Pillow Image Rectangle box
# (x1, y1) (left, top)
# (right, bottom) (x2, y2)

# (top,right,bottom,left)
# (32,64,0,0)

#RESIZE
def resize_image(image,w,h):
    image=cv2.resize(image,(w,h))
    return image

#crop
def crop_image(image,y1,y2,x1,x2):
    image=image[y1:y2,x1:x2]
    return image

def padding_image(image,topBorder,bottomBorder,leftBorder,rightBorder,color_of_border=[0,0,0]):
    image = cv2.copyMakeBorder(image,topBorder,bottomBorder,leftBorder,
        rightBorder,cv2.BORDER_CONSTANT,value=color_of_border)
    return image

def flip_image(image,dir):
    image = cv2.flip(image, dir)
    return image

def add_light(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        return image
    else:
        return image

def saturation_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def scale_image(image,fx,fy):
    image = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
    return image

def translation_image(image,x,y):
    rows, cols ,c= image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def rotate_image(image,deg):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image