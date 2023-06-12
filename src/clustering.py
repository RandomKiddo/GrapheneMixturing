import cv2
import numpy as np
from sklearn.cluster import MeanShift, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import math

def process(fp: str) -> None:
    """
    Pre-process an image to be edge detected and clustered
    :param fp: The image filepath
    :return: None
    """

    # Start timing the process
    start_time = time.time()

    # Open the image, downsize the features to 100x75 and store the image's shape and
    # a 2D copy of the image of shape (7500, 3)
    img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 75))
    img_flattened = img.reshape((-1, 3))
    shape = img.shape
    w, h = img.shape[:2]

    # The next few lines figures out the most occurring / dominant color of the image,
    # which will be the background normalization fill image. We use a KMeans sequence
    # to find 5 most occurring colors (variable n_colors). We then save the fill color
    # to the variable fill
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    fill = dominant.tolist()

    # Create a grayscale version of the target image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Plot the image and a histogram of the grayscale color values
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    hist, bin = np.histogram(gray.ravel(), 256, [0, 255])
    plt.xlim([0, 255])
    plt.plot(hist)
    plt.title('histogram')
    plt.show()

    # Get the max value of the histogram and define left and right thresholds for the grayscale
    # color values. We don't want anything lighter than the substrate background, so we take a
    # small right threshold. For the left threshold, we use the rgb to grayscale formula to deduce
    # a value: Y = 0.299r + 0.587g + 0.114b. If the substrate has background of (r0, g0, b0) and
    # the sample has color (r0 +/- α, g0 - 5l, b0 +/- β), where α, β take into account differences
    # in the red and green values, and l is the amount of layers of the sample, then by using the
    # grayscale formula, we get that the difference in grayscale values between the substrate and
    # the sample would be +/- α +/- β - 2.935l. Since we have l ∈ [1, 5], the upper bound of the
    # difference, taking an α ≈ β ≈ 2, we get a right threshold of about 20.
    # However, many images have contrast differences, so we increase the left threshold to a
    # value of 35 (this helps deal with contrast differences as well).
    max_value = np.argmax(hist)
    threshold_left = 35
    threshold_right = 1

    # Apply the pixels outside the thresholds to the fill color defined prior.
    w, h = gray.shape[:2]
    new = img.copy()
    for r in range(w):
        for c in range(h):
            if gray[r, c] < max_value - threshold_left or gray[r, c] > max_value + threshold_right:
                new[r, c] = fill
            else:
                new[r, c] = img[r, c]

    # Median blur the image to remove stray lines, and normalize the background.
    # We also go through a bilateral filter round before and after the median blur
    # to help preserve the edges of the image.
    new = cv2.bilateralFilter(new, d=5, sigmaColor=10, sigmaSpace=1)
    new = cv2.bilateralFilter(new, d=5, sigmaColor=5, sigmaSpace=1)
    new = cv2.bilateralFilter(new, d=5, sigmaColor=1, sigmaSpace=1)
    new = cv2.medianBlur(new, 3)
    new = cv2.bilateralFilter(new, d=5, sigmaColor=2, sigmaSpace=1)
    new = cv2.bilateralFilter(new, d=5, sigmaColor=1, sigmaSpace=1)
    new = cv2.bilateralFilter(new, d=5, sigmaColor=0.5, sigmaSpace=1)
    plt.imshow(new)
    plt.show()

    # Normalize the pixels of the new image, and create a flattened copy of the
    # image of shape (7500, 3)
    new = new / 255
    new_flattened = new.reshape((-1, 3))

    # Initialize a sklearn MeanShift algorithm to help identify central clusters
    # of color on the substrate. We choose not to cluster every single point by
    # setting cluster_all=False. We fit the flattened image to the MeanShift instance.
    # Then we retrieve the labels and cluster centers of the image.
    ms = MeanShift(cluster_all=False)
    ms.fit(new_flattened)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_clusters_ = len(labels)

    # Plot a 3D graph of where the cluster centers are in RGB colorspace.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
               marker='x', c='#ff0000', linewidths=2.5)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()

    # Calculate the distances between each center and its other centers
    dists = []
    for i in range(len(cluster_centers)):
        row = []
        for j in range(len(cluster_centers)):
            u = (cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2])
            v = (cluster_centers[j, 0], cluster_centers[j, 1], cluster_centers[j, 2])
            dist = distance(u, v)
            row.append(dist)
        dists.append(row)

    # Define a normalized threshold distance of 0.1, and remove any falsy cluster centers
    # by checking if the average distance it is from other cluster centers is "too large",
    # i.e. above the threshold value.
    threshold = .1
    remove = []
    for _ in range(len(dists)):
        avg = sum(dists[_])/len(dists[_])
        if abs(avg) >= threshold:
            remove.append(cluster_centers[_])

    # Display the mean shift image, and replace the nearby pixels with the
    # dominant background color if it is "nearby" a cluster center to remove
    ms_img = new.copy()
    for r in range(w):
        for c in range(h):
            u = new[r, c]
            found = False
            for _ in remove:
                if nearby(u, _):
                    ms_img[r, c] = (fill[0] / 255, fill[1] / 255, fill[2] / 255)
                    found = True
                    break
            if not found:
                ms_img[r, c] = u
    plt.imshow(ms_img)
    plt.show()

    # Show a 3D subplot of the image RGB values normalized before and
    # after the MeanShift implementation.
    pixel_colors = img.reshape((np.shape(new)[0] * np.shape(new)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    r, g, b = cv2.split(new)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()

    pixel_colors = img.reshape((np.shape(ms_img)[0] * np.shape(ms_img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    r, g, b = cv2.split(ms_img)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()

    # Another round of bilateral filtering
    img_bl = ms_img.copy()
    img_bl = img_bl * 255.0
    img_bl = np.uint8(img_bl)
    img_bl = cv2.bilateralFilter(img_bl, d=5, sigmaColor=0.5, sigmaSpace=2)
    img_bl = cv2.bilateralFilter(img_bl, d=5, sigmaColor=0.25, sigmaSpace=2)
    img_bl = cv2.bilateralFilter(img_bl, d=5, sigmaColor=0.25, sigmaSpace=1)
    img_bl = cv2.bilateralFilter(img_bl, d=5, sigmaColor=0.5, sigmaSpace=1)
    plt.imshow(img_bl)
    plt.show()

    # todo clahe boost or try Lapaclian

    # Grayscale the image and apply Sobel edge detection to the image to find
    # sample edges.
    img_bl = cv2.cvtColor(img_bl, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(img_bl, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_bl, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Define a lower threshold of 8.0 in grayscale space, and use that to produce
    # a binary mask image of 1s and 0s
    threshold = 8.0
    grad2 = grad.copy()
    for r in range(w):
        for c in range(h):
            if grad[r, c] < threshold:
                grad2[r, c] = 0
            else:
                grad2[r, c] = 1
    plt.imshow(grad2, cmap='gray')
    plt.show()

    # Floodfill the mask (i.e. fill in the space between the edges). Since the floodFill
    # method requires a mask of size (rows+2, cols+2, 1), we later crop the image back
    # to (75, 100, 1)
    mask = np.zeros((77, 102, 1), np.uint8)
    cv2.floodFill(grad2, mask, (0, 0), 1)
    mask = mask[0:75, 0:100]

    # The mask's 1s and 0s need to be inverted. This can be done by taking the new mask
    # and setting the value to 1 - the old mask value
    mask_inv = mask.copy()
    for r in range(w):
        for c in range(h):
            mask_inv[r, c] = 1 - mask[r, c]

    # Display the regions of interest (RoI)
    plt.imshow(cv2.bitwise_and(img, img, mask=mask_inv))
    plt.show()

    # Finish the process and calculate the time required for the process
    close_time = time.time()
    print('Processed in --- %s seconds' % (close_time - start_time))

def distance(u: tuple, v: tuple) -> float:
    return math.sqrt(
        ((u[0]-v[0])**2) + ((u[1]-v[1])**2) + ((u[2]-v[2])**2)
    )

def nearby(point: tuple, center: tuple, eps: float = 0.01) -> bool:
    dist = distance(point, center)
    return abs(dist) <= eps


if __name__ == '__main__':
    process(fp='/Users/firsttry/Desktop/Lab/test/3.jpg')

