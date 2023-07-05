import statistics
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from typing import *
import time
from sklearn.mixture import GaussianMixture

def process(fp: str, verbose: Optional[bool] = False, clean: Optional[bool] = True, timed: Optional[bool] = True,
            intense_clean: Optional[bool] = True, console_output: Optional[bool] = False) -> Tuple[Any, Any, float]:
    """
    Processes a sample image
    :param fp: <str> The image filepath
    :param verbose: <Optional[bool]> If all images should be shown
    :param clean: <Optional[bool]> If the processed image should be cleaned
    :param timed: <Optional[bool]> If the process should be timed
    :param intense_clean: <Optional[bool]> If the processed image should be cleaned through R^3 distances
    :param console_output: <Optional[bool]> If any console logging should be outputted to the console
    :return: <Tuple[Any, Any, float]> The original image, the processed one, and the runtime (if timed)
    """

    if console_output:
        print('------------------------------------------------------')
        print('--- Begin Photo Processing ---')

    # If timing, start the timer
    if timed:
        start = time.time()
    else:
        start = 0

    # Read the image, covert to LAB, and resize to 75x100
    img = cv2.imread(fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = cv2.resize(img, (100, 75))
    w, h = img.shape[:2]
    if console_output:
        print('--- Image Opened ---')

    # Filter out any L values that not in [100, 150]
    filtered = img.copy()
    for r in range(w):
        for c in range(h):
            if not 100 <= img[r, c][0] <= 150:
                filtered[r, c] = (0, 0, 0)
            else:
                filtered[r, c] = img[r, c]
    if console_output:
        print('--- Image Filtered ---')

    # Extract background LAB color from filtered image, then set the
    # filtered pixels to the background color
    l = a = b = total = 0
    R = G = B = total_rgb = 0
    img_rgb = cv2.resize(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB), (100, 75))
    for r in range(w):
        for c in range(h):
            if tuple(filtered[r, c]) != (0, 0, 0):
                l += filtered[r, c][0]
                a += filtered[r, c][1]
                b += filtered[r, c][2]
                total += 1
                R += img_rgb[r, c][0]
                G += img_rgb[r, c][1]
                B += img_rgb[r, c][2]
                total_rgb += 1
    bg = (l/total, a/total, b/total)
    bg_rgb = (R/total_rgb, G/total_rgb, B/total_rgb)
    for r in range(w):
        for c in range(h):
            if tuple(filtered[r, c]) == (0, 0, 0):
                filtered[r, c] = bg
    if console_output:
        print('--- Background Extracted ---')

    # Median blur the image using a kernel size of 3
    img_blur = cv2.medianBlur(filtered, 3)
    if verbose:
        plt.imshow(img_blur)
        plt.show()

    # Isolate the L values of the blurred image as a grayscale-esque image
    l_img = np.zeros((75, 100, 1), dtype=np.int32)
    for r in range(w):
        for c in range(h):
            l, a, b = img_blur[r, c]
            l_img[r, c] = l
    if verbose:
        plt.imshow(l_img)
        plt.show()
    if console_output:
        print('--- L-Img Created ---')

    # Use an Otsu Thresholding on the L image. We then use a value sigma=0.33, and use the threshold
    # to define a Canny range of [(1-sigma)*threshold, (1+sigma)*threshold]. For the Canny image, we
    # use an aperture size of 5, and we do use the L2gradient
    threshold, _ = cv2.threshold(np.uint8(l_img), 0, 255, cv2.THRESH_OTSU)
    sigma = 0.33
    c_img = cv2.Canny(np.uint8(l_img), (1-sigma)*threshold, (1+sigma)*threshold, apertureSize=5, L2gradient=True)
    if verbose:
        plt.imshow(c_img)
        plt.show()
    if console_output:
        print('--- Otsu-Sigma Thresholding w/ Canny Finalized ---')

    # We wish to fill in any gaps in the image, so we define a kernel of size 3x3 of ones. We then
    # dilate the canny image for one iteration with the kernel and then successively erode the
    # image back down from the dilated image using the same kernel in one iteration
    kernel = np.ones((3, 3), np.uint8)
    d_im = cv2.dilate(c_img, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1)
    if console_output:
        print('--- Dilation and Erosion Completed ---')

    # We use the eroded image to find the contours using RETR_EXTERNAL and
    # CHAIN_APPROX_SIMPLE. We the find the main contour (0 if only 2
    # contours, 1 otherwise), and then max the contour using cv2.contourArea
    contours = cv2.findContours(e_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    if console_output:
        print('--- Contours Filled ---')

    # We create a zeros_like copy of the image and draw the contours
    result = np.zeros_like(img)
    cv2.drawContours(result, [big_contour], 0, (255, 255, 255), cv2.FILLED)
    if verbose:
        plt.imshow(result, cmap='gray')
        plt.show()

    # We prepare for clustering by reading another copy of the image, but this time into
    # RGB instead of LAB. We set any background image pixels to black- rgb(0, 0, 0)
    gmm_prep = cv2.resize(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB), (100, 75))
    for r in range(w):
        for c in range(h):
            if tuple(result[r, c]) != (255, 255, 255) or c_img[r, c] == 1:
                gmm_prep[r, c] = bg_rgb
    if verbose:
        plt.imshow(gmm_prep)
        plt.show()

    # We use DBSCAN to cluster the image. We use an epsilon of 1.4, and we use a min_samples
    # values of channels+1 = 3+1 = 4. We fit the instance to the prepared data, flattened from
    # 75x100x3 to 7500x3, and then retrieve the labels. If we are not cleaning the image,
    # then we show the image and complete the program.
    db = DBSCAN(eps=1.4, min_samples=4)
    db.fit(gmm_prep.reshape((-1, 3)))
    labels = db.labels_
    if console_output:
        print('--- DBSCAN and GMM Prep Completed ---')

    # We then use GaussianMixture modeling on 5 components to help reduce the number of features
    # provided by the DBSCAN (somtimes upwards of 40!). We are required to expand the dimensions
    # of the labels from the DBSCAN for this to operate.
    N = 0
    for _ in range(2, 26):
        gmm = GaussianMixture(n_components=_, covariance_type='tied')
        gmm.fit(np.expand_dims(labels, axis=-1))
        value = gmm.bic(np.expand_dims(labels, axis=-1))
        if value < 0:
            N = _-1
            break
    if console_output:
        print(f'--- N Components Selected; Using N={N} ---')

    gmm = GaussianMixture(n_components=N, covariance_type='tied')
    gmm.fit(np.expand_dims(labels, axis=-1))
    gmm_labels = gmm.predict(np.expand_dims(labels, axis=-1))
    if console_output:
        print('--- GMM_N Completed ---')

    # True by default; Clean up the images to reduce noise. This not only deals with stray pixels
    # from the DBSCAN and GaussianMixture algorithms (through kernel size 3 median blurring), but
    # by also clearing out any pixels that are "stray". A stray pixel is defined as the following
    # in our specific scenario: Any pixel not labeled as the background label (the mode), that
    # is not contained within the ORIGINAL Canny image (pre-dilation and pre-erosion). We check this
    # through a simple algorithm:
    # 1. Check if the pixel is not a background pixel. If no, move to the next pixel
    # 2. Starting from the pixel, go leftwards and see if there is a Canny detected pixel
    # 3. Do the same for the right side
    # 4. If the pixel is enclosed in the left and right, we can assume it's not stray
    # This algorithm is not perfect, but works well for pixels near the sample.
    if clean:
        if console_output:
            print('--- Cleaning Detected ---')
        blur = cv2.medianBlur(np.uint8(gmm_labels.reshape((75, 100))), 3)
        mode = statistics.mode(gmm_labels)
        for r in range(w):
            for c in range(h):
                if blur[r, c] != mode:
                    found_left = found_right = False
                    for _ in range(c, -1, -1):
                        try:
                            if c_img[r, _] != 0:
                                found_left = True
                        except IndexError:
                            continue
                    for _ in range(c, 100, 1):
                        try:
                            if c_img[r, _] != 0:
                                found_right = True
                        except IndexError:
                            continue
                    if not found_left or not found_right:
                        blur[r, c] = mode
        if console_output:
            print('--- Stray Pixels Adjusted ---')

        # If intense cleaning is detected (True by default) we continue cleaning using the following
        # methodology:
        # 1. Collect the color RGB values by label
        # 2. Check each pixel with it's R^3 distance to other RGB values of its class (by this we mean
        # how close an image is in color to its others).
        # 3. If the image is too far (pre-defined using 5%), we stage it for reclassification
        # 4. Iterate each pixel needing to be reclassified and compare it to all other classes. Then
        # select the best class for the pixel
        if intense_clean:
            if console_output:
                print('--- Intense Cleaning Detected ---')
            colors = {}
            for r in range(w):
                for c in range(h):
                    R, G, B = img[r, c]
                    label = blur[r, c]
                    if colors.get(label) is None:
                        colors[label] = [(R, G, B), ]
                    else:
                        colors[label].append((R, G, B))
            reclassify = []
            for r in range(w):
                for c in range(h):
                    R, G, B = img[r, c]
                    label = blur[r, c]
                    proximity = []
                    if label == mode:
                        continue
                    for _ in colors[label]:
                        proximity.append(distance([R, G, B], list(_)))
                    avg = sum(proximity)/len(proximity)
                    if avg > .05:
                        reclassify.append(((R, G, B), r, c, label))
            for data in reclassify:
                averages = {}
                for label in colors:
                    proximity = []
                    if data[3] == label:
                        continue
                    for _ in colors[label]:
                        proximity.append(distance(list(data[0]), list(_)))
                    avg = sum(proximity)/len(proximity)
                    averages[label] = avg
                closest = 10000000  # some large value
                label_closest = None
                for _ in averages:
                    if averages[_] < closest:
                        closest = averages[_]
                        label_closest = _
                blur[data[1], data[2]] = label_closest
            if console_output:
                print(f'--- {len(reclassify)} Pixels Reclassified ---')
        if console_output:
            print('--- Image Cleaned ---')
    else:
        blur = gmm_labels.reshape((75, 100))

    # Show a side-by-side plot of the given image and the clustered result
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(blur)
    plt.title('Clustered Image')
    plt.show()

    # If timing, stop the timer
    if timed:
        end = time.time()
        process_time = end - start
    else:
        process_time = 0

    if console_output:
        print(f'--- Photo Processed In ~{int(round(process_time, 0))} Seconds ---')
        print('------------------------------------------------------')

    # Return a copy of the image, and the clustered image labels
    return img, labels, process_time

def distance(u: list, v: list) -> float:
    """
    Calculates the R^3 distance between the two pixels provided, mathematically defined as ||u-v||,
    or simply sqrt((u1-v1)^2 + (u2-v2)^2 + (u3-v3)^2)
    :param u: The first pixel RGB
    :param v: The second pixel RGB
    :return: A float of the value of ||u-v||
    """
    u[0] /= 255
    u[1] /= 255
    u[2] /= 255
    v[0] /= 255
    v[1] /= 255
    v[2] /= 255
    return (((u[0]-v[0])**2)+((u[1]-v[1])**2)+((u[2]-v[2])**2))**0.5


if __name__ == '__main__':
    _, _, _ = process(fp='/Users/firsttry/Desktop/Lab/test/3.png', console_output=True)
    _, _, _ = process(fp='/Users/firsttry/Desktop/Lab/test/3.jpg', console_output=True)
