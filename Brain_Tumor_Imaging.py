# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:47:35 2022

@author: brian
"""


import numpy as np
import matplotlib.pyplot as plt


"""
Function init + test examples
"""
#images used to draw conclusions Y1.jpg, Y167.jpg, Y147.jpg, Y22.jpg, Y47.jpg,
#Y251.jpg, Y159.jpg,Y73.jpg, Y46.jpg
path  = r'C:\Users\brian\OneDrive\Desktop\MI\Proiect MI\yes\Y46.jpg'
img = plt.imread(path)

plt.figure()
plt.imshow(img, cmap = 'gray'),plt.title('Original_Test')

s = img.shape
print(s)

"""
Grayscaling

"""

def rgb2gray(img_in, tip):
    s =img.shape
    img_in = img_in.astype('float')
    if len(s) == 3 and s[2] ==3:
        if tip == 'png':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2]) * 255
        elif tip == 'jpg':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2])
        img_out = np.clip(img_out,0,255)    
        img_out = img_out.astype('uint8')
        return img_out
    else:
        print('the image is not a color image')
        
    
"""
Test

"""    

s = img.shape        
img_out = np.zeros((s[0],s[1]), dtype = 'uint8')
img_out = rgb2gray(img, 'jpg')        
plt.figure()
plt.imshow(img_out, cmap = 'gray'),plt.title('Grayscale_Test')

img = img_out


"""
Linear transformation

"""

def linear_transform(img, a, b, Ta, Tb):
     s = img.shape
     img_out = np.zeros((s[0], s[1]), dtype='uint8')
     
     for i in range(s[0]):
         for j in range(s[1]):
             if img[i,j] < a:
                 img_out[i,j] = (Ta/a) * img[i,j]
             elif img[i,j] >= a and img[i,j] <=b:
                 img_out[i,j] = Ta + (Tb-Ta)*(img[i,j]-a)/(b-a)
             else:
                 img_out[i,j] = Tb + (255-Tb)*(img[i,j]-b)/(255-b) 
     return img_out
 
     
"""

Test

"""     

img_linear = linear_transform(img, 100, 200, 20, 200)
plt.figure()
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(img_linear, cmap='gray'), plt.title('Linearization_Test')

"""
Binarization

""" 

def binary_transform(img, T):
     s = img.shape
     img_out = np.zeros((s[0], s[1]), dtype='uint8')
     
     for i in range(s[0]):
        for j in range(s[1]):
             if img[i,j] < T:
                 img_out[i,j] = 0
             else:
                 img_out[i,j] = 255
     return img_out
 
"""

Test

"""     
 
img_binar = binary_transform(img, 160)
plt.figure()
plt.subplot(121); plt.imshow(img, cmap = 'gray'); plt.title('Original image') 
plt.subplot(122); plt.imshow(img_binar, cmap = 'gray'); plt.title('Binarization_Test')


"""

Exponential Function

""" 

def exponential(img):
     s = img.shape
     img = img.astype('float')
     img_out = np.zeros((s[0], s[1]), dtype='uint8')
#     
     img_out = 256 ** (img/255) - 1
     return img_out   
 
"""

Test

"""     
 
img_exp = exponential(img) 
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray'); plt.title('Original image') 
plt.subplot(1,2,2); plt.imshow(img_exp, cmap = 'gray'); plt.title('Exponential_Test')


"""

Parameters for mathematical morphology functions

""" 

se1 = np.ones((8,5))

#afisarea continutului variabilei se1
print('se1 =', se1)

# define another structurant element: V4 si V8
V4 = np.array([[1,0,0],[0,1,0],[0,0,1]])
V8 = np.ones((3,3))

print('V4 =', V4)
print('V8 =', V8)


import scipy.ndimage.morphology as morpho


"""

Erosion + Test

"""  

# erodarea imaginii folosind elementul structurant se1
er1 = morpho.binary_erosion(img_binar, se1)
plt.imshow(er1, cmap = 'gray') 

plt.figure()
plt.subplot(1,2,1), plt.imshow(img_binar, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(er1, cmap = 'gray'),plt.title('erodata_test') 

"""

Opening + Test

"""  

se4 = np.ones((20,20))
opened = morpho.binary_opening(img_binar,se4)
plt.figure()
plt.subplot(1,2,1), plt.imshow(img_binar, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(opened, cmap = 'gray'),plt.title('opened_test')

