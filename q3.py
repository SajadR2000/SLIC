import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
# from time import time


def initial_cluster_centers_finder(img_rgb, K):
    """
    This function takes an image and uniformly chooses K samples from it and then moves them to a pixel with the
    smallest gradient
    :param img_rgb: input image
    :param K: number of samples
    :return: samples xy and lab
    """
    # Shape of the image:
    h, w, n_ch = img_rgb.shape
    kx = int(np.round(np.sqrt(w / h * K)))
    ky = int(np.round(np.sqrt(h / w * K)))
    # print(kx, ky)
    # print(ky * kx)
    dx = w // (kx + 1)
    dy = h // (ky + 1)
    cx_vec = np.arange(dx, (kx + 1) * dx, dx)
    # print(cx_vec)
    cy_vec = np.arange(dy, (ky + 1) * dy, dy).reshape((-1, 1))
    cx_vec = np.repeat(cx_vec, ky)
    cy_vec = np.repeat(cy_vec, kx, axis=1).T.reshape((-1,))
    # print(cx_vec)
    # print(cy_vec)
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    grad_filter_y = grad_filter_x.T
    # Important: use float as the output. because some elements may be negative. For example -1 would be 255.
    grad_x = cv2.filter2D(gray_img, cv2.CV_64F, grad_filter_x)
    grad_y = cv2.filter2D(gray_img, cv2.CV_64F, grad_filter_y)
    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))

    half_window_size = 2  # Window size w = 2 * half_window_size + 1
    centers_lab = np.zeros((len(cx_vec), 1, 3), dtype=np.uint8)
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    for cx, cy, i in zip(cx_vec, cy_vec, np.arange(len(cx_vec))):
        x_st = np.max((0, cx - half_window_size))
        y_st = np.max((0, cy - half_window_size))
        x_end = np.min((w, cx + half_window_size + 1))
        y_end = np.min((w, cy + half_window_size + 1))

        window = grad[y_st:y_end, x_st:x_end]
        new_center = np.argwhere(window == np.min(window))[0]
        cx_vec[i] = x_st + new_center[1]
        cy_vec[i] = y_st + new_center[0]
        centers_lab[i, 0, :] = lab_img[cy_vec[i], cx_vec[i], :]

    centers_yx = np.array([cy_vec, cx_vec]).T
    return centers_yx, centers_lab


def SLIC(rgb_img, K=2048, alpha=0.02, tolerance=0.0001, max_iter=20):
    """
    This function takes an image and the number of desired superpixels and runs SLIC algorithm to find them.
    :param rgb_img: Input image in RGB color-space
    :param K: Number of desired superpixels
    :param alpha: Controls tradeoff between spatial (xy) difference and color (Lab) difference
    :param tolerance: Metric of convergence
    :param max_iter: maximum number of SLIC iterations
    :return: label image
    """
    # Convert input image to Lab color-space
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
    # Find initial cluster centers (cluster = superpixel)
    initial_centers_yx, initial_centers_lab = initial_cluster_centers_finder(rgb_img, K)
    # The number of superpixels equals K?
    # assert K == initial_centers_yx.shape[0]
    K = initial_centers_yx.shape[0]
    # print(K)
    # Sx = w / (sqrt(K) + 1), Sy = h / (sqrt(K) + 1) --> so approximately set S = max(w / sqrt(K), h / sqrt(K))
    h, w, n_ch = rgb_img.shape
    S = int(np.ceil(np.max((w / np.sqrt(K), h / np.sqrt(K)))))
    # Creating the grid, assign to each pixel a xy. Note that xys are the indices.
    # Example: Consider a 3*3 img:
    # cols = [0 1 2
    #         0 1 2
    #         0 1 2]
    # rows = [0 0 0
    #         1 1 1
    #         2 2 2]
    cols = np.repeat(np.arange(w).reshape((1, -1)), h, axis=0)
    rows = np.repeat(np.arange(h).reshape((-1, 1)), w, axis=1)
    # Total cost of ij pixel when assigned to clusters[i, i] = D_total[i, j]
    # Set the initial cost as infinite
    # Set the initial clusters as -1
    D_total = np.inf * np.ones((h, w))
    clusters = np.zeros((h, w), dtype=int)
    for _ in range(max_iter):
        # t = time()
        # plt.scatter(initial_centers_yx[:, 0], initial_centers_yx[:, 1])
        # plt.show()
        for i, loc in zip(np.arange(K), initial_centers_yx):
            # Set the starting and ending points of the 2S*2S square for each cluster
            # Note that these indices must be greater or equal to zero and less than or equal to w,h.
            x_st = np.max((loc[1] - S, 0))
            y_st = np.max((loc[0] - S, 0))
            x_end = np.min((loc[1] + S + 1, w))
            y_end = np.min((loc[0] + S + 1, h))
            # Find the Lab of the pixels in square window
            window_lab = lab_img[y_st:y_end, x_st:x_end, :]
            # Lab of the i-th superpixel center
            center_lab = initial_centers_lab[i, 0, :]
            # d_lab
            dlab = np.sum(np.square(window_lab - center_lab), axis=2)
            # d_xy
            dx = np.square(cols[y_st:y_end, x_st:x_end] - loc[1])  # / S: uncomment to scale the xys
            dy = np.square(rows[y_st:y_end, x_st:x_end] - loc[0])  # / S
            dxy = dx + dy
            # d = d_lab + alpha * d_xy
            d = dlab + alpha * dxy
            # Find the current cost of each pixel with its assigned superpixel center
            D_window = D_total[y_st:y_end, x_st:x_end]
            # For each pixel in the square window check whether if i-th cluster is assigned to it, its cost reduces
            update_condition = d < D_window
            # If so, update the cost and the clusters assigned to pixels satisfying the update condition
            D_window[update_condition] = d[update_condition]
            D_total[y_st:y_end, x_st:x_end] = D_window.copy()
            cluster_window = clusters[y_st:y_end, x_st:x_end]
            cluster_window[update_condition] = i
            clusters[y_st:y_end, x_st:x_end] = cluster_window.copy()
        # print(time() - t)

        # Store the centers before updating them to check the convergence criterion.
        prev_centers_yx = initial_centers_yx.copy()
        prev_centers_lab = initial_centers_lab.copy()
        # t = time()
        for i, loc in zip(np.arange(K), initial_centers_yx):
            # 2S*2S square corners
            x_st = np.max((loc[1] - S, 0))
            y_st = np.max((loc[0] - S, 0))
            x_end = np.min((loc[1] + S + 1, w))
            y_end = np.min((loc[0] + S + 1, h))
            # For each cluster point check the window and find the pixels having assigned to it.
            # Note that there is no need to search the whole image due to the logic of previous part.
            # This considerably reduced runtime.
            cluster_window = clusters[y_st:y_end, x_st:x_end]
            # For each pixel in the window check whether it's assigned to i-th superpixel
            condition = cluster_window == i
            # Find the coordinate of pixels in the window
            cols_window = cols[y_st:y_end, x_st:x_end]
            rows_window = rows[y_st:y_end, x_st:x_end]
            # Update the i-th cluster mean as the mean of the pixels assigned to it
            x_mean = int(np.mean(cols_window[condition]))
            y_mean = int(np.mean(rows_window[condition]))
            initial_centers_yx[i, :] = [y_mean, x_mean]
            initial_centers_lab[i, 0, :] = lab_img[y_mean, x_mean]
        # print(time() - t)
        # Compute the sum residuals, if it's less than tolerance, algorithm has converged
        E = np.sum(np.square(initial_centers_yx - prev_centers_yx))
        E += np.sum(np.square(initial_centers_lab - prev_centers_lab))
        E = np.sqrt(E)
        # print(E)
        if E < tolerance:
            break
    return clusters


def run(input_img, K=2048, output_dir='./slic.jpg'):
    rgb_img = input_img.copy()
    cluster_labels = SLIC(rgb_img.copy(), K)
    # Removing outliers by median filter --> connected clusters
    cluster_labels = cv2.medianBlur(cluster_labels.astype(np.uint8), 25)
    # print(cluster_labels)
    # print(rgb_img)
    rgb_img = mark_boundaries(rgb_img, cluster_labels, (1, 0, 0))
    # print(rgb_img)
    plt.imsave(output_dir, rgb_img)


img = cv2.imread("./slic.jpg", cv2.IMREAD_UNCHANGED)
if img is None:
    print("Couldn't read the image!!")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)
run(img, 64, 'res06.jpg')
run(img, 256, 'res07.jpg')
run(img, 1024, 'res08.jpg')
run(img, 2048, 'res09.jpg')
