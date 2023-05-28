import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from pylab import array, plot, show, axis, arange, figure, uint8
import time
from typing import *
import statistics
from PIL import Image

# Time the process
instantiation_time = time.time()

# Path to the photos
path = '/Users/firsttry/Desktop/test/'

# Read the first photo and convert from CV2 BGR to RGB
img = cv2.imread(path + 'photo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Gaussian Blur the image by a kernel size of (33, 33)
img_gs = cv2.GaussianBlur(img, (33, 33), 0, 0)
img_gs = cv2.cvtColor(img_gs, cv2.COLOR_BGR2RGB)
cv2.imwrite(path + '1.png', img_gs)

# To preserve some of the edges after Gaussian Blurring
# We use a bilateral filter of d=5 (for real-time application)
# plus sigma values of 10 and 3 to the image
img_gs_bl = cv2.bilateralFilter(img_gs, 5, 10, 3)
cv2.imwrite(path + '2.png', img_gs_bl)

# Resize the image to 400x300 (if necessary)
resize = cv2.resize(img_gs_bl, (400, 300)) # todo add size check
cv2.imwrite(path + '3.png', resize)

# Declare variables for the contrast increase
# max_i = Max Intensity, phi and theta are parameters
# We found through testing that phi=theta=3 was the best
# resulting parameters for this program
max_i = 255.0
x = arange(max_i)
phi = 3
theta = 3

'''
# Increase intensity such that bright pixels only become
# mildly brighter while dark pixels become much brighter
nimg = (max_i/phi)*(resize/(max_i/theta))**0.5
nimg = array(nimg, dtype=uint8)

# Now using the intensified image, do a smiliar transformation
# where dark pixels become much darker now but bright pixels
# only become slightly darker. This step does eventually create
# false positives due to the color shift
y = (max_i/phi)*(x/(max_i/theta))**0.5
nimg1 = (max_i/phi)*(resize/(max_i/theta))**2
nimg1 = array(nimg1,dtype=uint8)
nimg1 = cv2.cvtColor(nimg1, cv2.COLOR_BGR2RGB)
cv2.imwrite(path + f'{phi}_{theta}_1.png', nimg1)

# Plot the figures
z = (max_i/phi)*(x/(max_i/theta))**2
figure()
plot(x,y,'r-') # Increased brightness
plot(x,x,'k:') # Original image
plot(x,z, 'b-') # Decreased brightness
axis('tight')
plt.savefig(path + f'{phi}_{theta}_2.png')
'''

clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
lab = cv2.cvtColor(resize, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
l2 = clahe.apply(l)
lab = cv2.merge((l2, a, b))
img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
cv2.imwrite(path + 'test.png', img2)

# Grayscale the image
gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# Apply Sobel image edge detection in the x and y
# Then convert the scale to return and absolute
# gradient
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# Combine the x and y absolute gradients to make the Sobel
# edge detection image, weighted at 50% for both directions
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imwrite(path + f'{phi}_{theta}_3.png', grad)

# Convert the edge detection image from RGB to LAB
img = cv2.imread(path + f'{phi}_{theta}_3.png')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_ch, a, b = cv2.split(lab)

# The following code enhances the edge detection brightness
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l_ch)

limg = cv2.merge((cl, a, b))
enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

cv2.imwrite(path + '0_1.png', enhanced)

img = enhanced
lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
l_ch, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l_ch)

limg = cv2.merge((cl, a, b))
enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

cv2.imwrite(path + '1_1.png', enhanced)

gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
cv2.imwrite(path + 'gray.png', gray)

# We then recieve the coordinates where the edges are detected
# And we create a binary image where a pixel is white if
# it is edge detected and black if not 
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
cv2.imwrite(path + 'binary.png', two)

# Retrieve the most occuring green value for the image
# This will later determine our layers
greens = []
for r in range(w):
    for c in range(h):
        R, G, B = resize[r, c]
        greens.append(G)
mode = max(set(greens), key=greens.count)
print(int(statistics.median(greens)))
print(np.mean(greens))
print(max(set(greens), key=greens.count))

# Define the color scale we will use based on the layers
resize = Image.open(path + '3.png')
final = resize.copy()
colors = {
    'nothing': (166, 77, 255), # lavender
    '5': (255, 0, 0), # red
    '4': (255, 128, 0), # orange
    '3': (255, 255, 0), # yellow
    '2': (0, 255, 85), # malachite
    '1': (0, 255, 255) # aqua
}

# Color the image based on the layers using green color
# values. We have not yet dealt with false positives
for r in range(resize.height):
    for c in range(resize.width):
        R, G, B = resize.getpixel((c, r))
        diff = mode - G
        if diff < 4:
            final.putpixel((c, r), colors['nothing'])
        elif diff < 9:
            final.putpixel((c, r), colors['1'])
        elif diff < 15:
            final.putpixel((c, r), colors['2'])
        elif diff < 20:
            final.putpixel((c, r), colors['3'])
        elif diff < 25:
            final.putpixel((c, r), colors['4'])
        elif diff < 30:
            final.putpixel((c, r), colors['5'])
        else:
            final.putpixel((c, r), colors['nothing'])
final.save(path + 'final.png')

# Overlay the edges onto the colored image
final_we = final.copy()
for r in range(resize.height):
    for c in range(resize.width):
        if (two[r, c] == (255, 255, 255)).all():
            final_we.putpixel((c, r), (0, 0, 0))
final_we.save(path + 'final_with_edges.png')

# Print program run time
close_time = time.time()
print('Photo Processed in --- %s seconds' % (close_time - instantiation_time))
