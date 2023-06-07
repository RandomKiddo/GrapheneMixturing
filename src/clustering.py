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
    start_time = time.time()

    img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 75))
    img_flattened = img.reshape((-1, 3))
    shape = img.shape

    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    fill = dominant.tolist()

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

    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    r, g, b = cv2.split(img)
    rgb = list(zip(r.flatten(), g.flatten(), b.flatten()))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('/Users/firsttry/Desktop/gray.jpg', gray)

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

    ms = MeanShift()
    ms.fit(gray.reshape(-1, 1))
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    cluster_centers = np.delete(cluster_centers, cluster_centers.argmin())
    cluster_centers = np.delete(cluster_centers, cluster_centers.argmax())

    minimum_cluster = min(cluster_centers)
    maximum_cluster = max(cluster_centers)

    cut = gray.copy()
    w, h = gray.shape[:2]
    for r in range(w):
        for c in range(h):
            if gray[r, c] < minimum_cluster or gray[r, c] > maximum_cluster:
                cut[r, c] = 0
            else:
                cut[r, c] = gray[r, c]

    new = img.copy()
    for r in range(w):
        for c in range(h):
            if cut[r, c] == 0:
                new[r, c] = fill
            else:
                new[r, c] = img[r, c]
    plt.imshow(new)
    plt.show()

    r, g, b = cv2.split(new / 256)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(b.flatten(), g.flatten(), r.flatten(), facecolors=pixel_colors)
    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')
    plt.show()

    new_flattened = (new / 255).reshape((-1, 3))

    gmm = GaussianMixture(n_components=5, covariance_type="tied")
    gmm = gmm.fit(new_flattened)

    cluster = gmm.predict(new_flattened)
    cluster = cluster.reshape(75, 100)
    plt.imshow(cluster)
    plt.show()

    close_time = time.time()
    print('Processed in --- %s seconds' % (close_time - start_time))


if __name__ == '__main__':
    process(fp='/Users/firsttry/Desktop/Lab/test/3.png')
