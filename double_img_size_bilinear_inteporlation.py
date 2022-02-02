import numpy as np
from PIL import Image
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')


def resize_image(name):
    img1 = Image.open(os.path.join(IMG_DIR, name))

    old = np.asarray(img1)  # convert to a numpy array
    rows, cols, layers = old.shape
    new = np.zeros((2*rows - 1, 2*cols - 1, layers))
    print(f'Original dimensions: {old.shape}')

    for layer in range(3):
        new[:, :, layer] = resize_layer(old[:, :, layer])

        # convert the values to usingned, 8-bit integers
        new = new.astype(np.uint8)
        print(f'        new dimensions: {new.shape}')

        img2 = Image.fromarray(new)  # convert back to Image
        newName = 'big_' + name
        img2.save(os.path.join(IMG_DIR, newName))


def resize_layer(old):
    rows, cols = old.shape

    # move old points
    rNew = 2*rows - 1
    cNew = 2*cols - 1
    new = np.zeros((rNew, cNew))
    new[0:rNew:2, 0:cNew:2] = old[0:rows, 0:cols]

    """ alternative approach
    # something like this would be necessary in languages
    # that don't support slicing
    new = np.zeros((2*rows - 1, 2*cols - 1))
    for r in range(rows):
        for c in range(cols):
            new[2*r, 2*c] = old[r, c]
    rows, cols = new.shape
    """

    # produce vertical values
    new[1:rNew:2, :] = (new[0:rNew-1:2, :] + new[2:rNew:2, :]) / 2
    """ alternative approach
    for r in range(1, rows, 2):
        for c in range(0, cols, 2):
            # top + bottom
            new[r, c] = (new[r-1, c] + new[r+1, c]) // 2  
    """
    # produce horizontal values
    new[:, 1:cNew:2] = (new[:, 0:cNew-1:2] + new[:, 2:cNew:2]) / 2
    """ alternative approach
    for r in range(0, rows, 2):
        for c in range(1, cols, 2):
            # left + right
            new[r, c] = (new[r, c-1] + new[r, c+1]) // 2
    """
    # produce center values
    new[1:rNew:2, 1:cNew:2] = (new[0:rNew-2:2, 0:cNew-2:2] +
                               new[0:rNew-2:2, 2:cNew:2] +
                               new[2:rNew:2, 2:cNew:2]) / 4
    """ alternative approach
    for r in range(1, rows, 2):
        for c in range(1, cols, 2):
            # top + bottom + left + right
            new[r, c] = (new[r-1, c] + new[r+1, c] + new[r, c-1] + new[r,c+1]) // 4
    """
    return new


if __name__ == '__main__':
    filename = 'img_siena.jpg'
    resize_image(filename)
