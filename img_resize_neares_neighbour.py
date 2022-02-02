# ==================================
# Import Libraries
# ==================================
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import os
from math import floor
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')

# ----------------------------------
# Euclidian Distance
# ==================================


def euclidian_dist(a, b):
    '''
    Euclidian distance between 2 points a(x_a, y_a) and b(x_b, y_b)
    Distance = Square Root ( (x_a - x_b)^2 +  (y_a - y_b)^2 )
    '''
    return np.sqrt(((a[0]-b[0])**2)+((a[1]-b[1])**2))

# ----------------------------------
# Nearest Neighbour
# ==================================


def near_neighbour(X, P):
    '''
    The nearest neighbour of point X(x,y) to the centroid P(x_p, y_p)
    The Neighbourhood is defined by the Upper-Left corner of the point X, which means 3 neighbours and the point X. 
    '''
    i, j = X[0], X[1]
    A = [[i, j], [i, j+1], [i+1, j], [i+1, j+1]]
    dist = [euclidian_dist(A[0], P), euclidian_dist(
        A[1], P), euclidian_dist(A[2], P), euclidian_dist(A[3], P)]
    minpos = dist.index(min(dist))
    return A[minpos]

# ----------------------------------
# Nearest Neighbour Interpolation
# ==================================


def NN_interpolation(im, scale_factor):
    '''
    Interpolation of the image im with scale factor scale_factor, using Nearest Neighbour.
    '''
    row, col = im.shape[0], im.shape[1]
    n_row, n_col = int(scale_factor * row), int(scale_factor * col)
    # fill in  img
    zoom = np.arange(n_row*n_col).reshape(n_row, n_col)
    print("zoom shape is: ", zoom.shape, "image shape is: ", im.shape, '\n')
    for i in range(n_row):
        for j in range(n_col):
            P = [floor(float(i)/scale_factor), floor(float(j)/scale_factor)]
            X = [int(i) for i in P]
            zoom[i][j] = im[near_neighbour(X, P)[0]][near_neighbour(X, P)[1]]
    return zoom

if __name__ == '__main__':      
    # -------------------------
    # Example
    # =========================
    im = imread(os.path.join(IMG_DIR, 'img_pare.jpg'))[..., 0]
    # im = imread(os.path.join(IMG_DIR, 'img_siena2.jpg'))[..., 0]

    J = NN_interpolation(im, 1.5)
    newImage = Image.fromarray(J)
    if newImage.mode != 'RGB':
        newImage = newImage.convert('RGB')
    newImage.save(os.path.join(IMG_DIR, 'NN_interpolation_img_pare.jpg'))

    # plt.figure(num='NN-Interpolation')
    # plt.subplot(121)
    # imgplot = plt.imshow(im, cmap="gray")  # Displaying the image
    # plt.title('Original')

    # plt.subplot(122)
    # imgplot = plt.imshow(J, cmap="gray")  # Displaying the image
    # plt.title('Zoomed')

    # plt.show()

    print(im.shape, J.shape)
