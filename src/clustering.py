import cv2
import numpy as np
from typing import *
from sklearn.cluster import MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import statistics

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

    # Plots a swatch of the dominant colors
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices] / float(counts.sum())]))
    rows = np.int_(img.shape[0] * freqs)

    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(dom_patch)
    ax.set_title('Dominant colors')
    ax.axis('off')
    plt.show()

    # Creates a colorscheme for the 3d scatter plot of the image colors
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # Gets the red, green, and blue channels of the image and then creates
    # a variable that stores the array of colors (r, g, b)
    r, g, b = cv2.split(img)
    rgb = list(zip(r.flatten(), g.flatten(), b.flatten()))

    # The 3d scatter plot of the image colors
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()

    # Create a grayscale version of the target image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Plot the image and a histogram of the grayscale color values
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    hist, bin = np.histogram(img.ravel(), 256, [0, 255])
    plt.xlim([0, 255])
    plt.plot(hist)
    plt.title('histogram')
    plt.show()

    # Instantiate a sklearn MeanShift sequence and fit it to the grayscale image.
    # This helps us gain clusters of the common color values. The cluster_all
    # field of the MeanShift is True by default, so all pixels will be clustered
    # in some sense
    ms = MeanShift()
    ms.fit(gray.reshape(-1, 1))
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    # Delete the far ends of the clusters
    cluster_centers = np.delete(cluster_centers, cluster_centers.argmin())
    cluster_centers = np.delete(cluster_centers, cluster_centers.argmax())

    # Find the maximum and minimum cluster grayscale values
    minimum_cluster = min(cluster_centers)
    maximum_cluster = max(cluster_centers)

    # Cut any pixel outside the range (minimum_cluster, maximum_cluster) as a
    # black pixel (grayscale value 0)
    cut = gray.copy()
    w, h = gray.shape[:2]
    for r in range(w):
        for c in range(h):
            if gray[r, c] < minimum_cluster or gray[r, c] > maximum_cluster:
                cut[r, c] = 0
            else:
                cut[r, c] = gray[r, c]

    # Do the same with a copy of the target image, but instead fill the colors outside
    # the range with the earlier defined background fill value. This helps normalize
    # the background color and get rid of tape residue, wafer particles, and more.
    # todo combine both for loops
    new = img.copy()
    for r in range(w):
        for c in range(h):
            if cut[r, c] == 0:
                new[r, c] = fill
            else:
                new[r, c] = img[r, c]
    plt.imshow(new)
    plt.show()

    # Split the new image and normalize it
    r, g, b = cv2.split(new / 256)

    # Plot the cut image rgb values in 3d colorspace like earlier
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(b.flatten(), g.flatten(), r.flatten(), facecolors=pixel_colors)
    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')
    plt.show()

    # Flatten the new image into (7500, 3) to prepare for Gaussian Mixture Modeling
    new_flattened = (new / 255).reshape((-1, 3))

    # Instantiate a sklearn GaussianMixture modeling sequence. We use covariance_type
    # of 'tied' and we wish to find n_components=5
    gmm = GaussianMixture(n_components=5, covariance_type="tied")
    gmm = gmm.fit(new_flattened)

    # Predict clusters on the data and then shape it back into the image format of (75, 100, 3)
    cluster = gmm.predict(new_flattened)
    cluster = cluster.reshape(75, 100)
    plt.imshow(cluster)
    plt.show()

    # Finish the process and calculate the time required for the process
    close_time = time.time()
    print('Processed in --- %s seconds' % (close_time - start_time))


if __name__ == '__main__':
    process(fp='/Users/firsttry/Desktop/Lab/test/3.png')
