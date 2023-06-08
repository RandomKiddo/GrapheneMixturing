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
    # However, many images have contrast differences, so we double the left
    # threshold to 40.
    max_value = np.argmax(hist)
    print(max_value)
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
    plt.imshow(new)
    plt.show()

    # Median blur the image to remove stray lines, and normalize the background.
    new = cv2.medianBlur(new, 3)
    plt.imshow(new)
    plt.show()

    # Flatten the image from (75, 100, 3) to (7500, 3)
    new_flattened = new.reshape((-1, 3))

    # Declare a GaussianMixture model from sklearn using covariance_type='tied' and
    # fit it to the flattened image, looking for n_components=2 (sample or no sample)
    gmm = GaussianMixture(n_components=2, covariance_type="tied")
    gmm = gmm.fit(new_flattened)

    # Predict clusters on the image, and reshape the result from (7500, 3) back to
    # (75, 100, 3), and display the image.
    cluster = gmm.predict(new_flattened)
    cluster = cluster.reshape(75, 100)
    plt.imshow(cluster)
    plt.show()

    # Finish the process and calculate the time required for the process
    close_time = time.time()
    print('Processed in --- %s seconds' % (close_time - start_time))


if __name__ == '__main__':
    process(fp='/Users/firsttry/Desktop/Lab/test/half.jpg')
    process(fp='/Users/firsttry/Desktop/Lab/test/max.jpg')
    process(fp='/Users/firsttry/Desktop/Lab/test/3.png')

