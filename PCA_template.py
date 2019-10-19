__author__ = 'schubert'

import matplotlib.image as npim
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import eigh


import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def svd_decomposition(img, vals):
    #TODO   

def pca_decomposition(img, vals):
    #TODO

img = npim.imread("bilder\katze.png")
gray = rgb2gray(img)
print(gray.shape )
Mhat1  = svd_decomposition(gray,20)
Mhat2 = svd_decomposition(gray,10)
Mhat3 = svd_decomposition(gray,5)



fig, [ax1, ax2, ax3,ax4] = plt.subplots(1, 4)
ax1.axis('off')
ax1.imshow(gray, cmap = plt.get_cmap('gray'))
ax1.set_title('true image')
ax2.axis('off')
ax2.imshow(Mhat, cmap = plt.get_cmap('gray'))
ax2.set_title('20 dims')
ax3.axis('off')
ax3.imshow(Mhat2, cmap = plt.get_cmap('gray'))
ax3.set_title('10 dims')
ax4.axis('off')
ax4.imshow(Mhat3, cmap = plt.get_cmap('gray'))
ax4.set_title('5 dims')
plt.show()
