import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from pylab import array, plot, show, axis, arange, figure, uint8 

path = '/Users/firsttry/Desktop/test/'

img = cv2.imread(path + 'photo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gs = cv2.GaussianBlur(img, (33, 33), 0, 0)
#plt.imshow(img_gs)
#plt.show()
img_gs = cv2.cvtColor(img_gs, cv2.COLOR_BGR2RGB)
cv2.imwrite(path + '1.jpg', img_gs)

img_gs_bl = cv2.bilateralFilter(img_gs, 5, 10, 3)
#plt.imshow(img_gs_bl)
#plt.show()
cv2.imwrite(path + '2.jpg', img_gs_bl)

resize = cv2.resize(img_gs_bl, (400, 300))
cv2.imwrite(path + '3.jpg', resize)

max_i = 255.0
x = arange(max_i)
phi = 3
theta = 3

nimg = (max_i/phi)*(resize/(max_i/theta))**0.5
nimg = array(nimg, dtype=uint8)

y = (max_i/phi)*(x/(max_i/theta))**0.5
nimg1 = (max_i/phi)*(resize/(max_i/theta))**2
nimg1 = array(nimg1,dtype=uint8)
nimg1 = cv2.cvtColor(nimg1, cv2.COLOR_BGR2RGB)
cv2.imwrite(path + f'{phi}_{theta}_1.jpg', nimg1)

z = (max_i/phi)*(x/(max_i/theta))**2
# Plot the figures
figure()
plot(x,y,'r-') # Increased brightness
plot(x,x,'k:') # Original image
plot(x,z, 'b-') # Decreased brightness
#axis('off')
axis('tight')
plt.savefig(path + f'{phi}_{theta}_2.jpg')

gray = cv2.cvtColor(nimg1, cv2.COLOR_RGB2GRAY)

grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imwrite(path + f'{phi}_{theta}_3.jpg', grad)

img = cv2.imread(path + f'{phi}_{theta}_3.jpg')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_ch, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l_ch)

limg = cv2.merge((cl, a, b))
enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

cv2.imwrite(path + '0_1.jpg', enhanced)

img = enhanced
lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
l_ch, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l_ch)

limg = cv2.merge((cl, a, b))
enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

cv2.imwrite(path + '1_1.jpg', enhanced)

# continue, use l_ch > 10
gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
cv2.imwrite(path + 'gray.jpg', gray)

indices = np.where(gray >= 50)
coordinates = list(zip(indices[0], indices[1]))
(w, h) = gray.shape[:2]
two = gray.copy()
two = cv2.cvtColor(two, cv2.COLOR_GRAY2RGB)
for r in range(w):
    for c in range(h):
        if (r, c) in coordinates:
            two[r, c] = [255, 255, 255]
        else:
            two[r, c] = [0, 0, 0]
cv2.imshow('two', two)
