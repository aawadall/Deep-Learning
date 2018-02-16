import numpy as np # linear algebra


def convolute(X, orig_x_size, orig_y_size, patch_x_size, patch_y_size, x_stride, y_stride):
    """Given input X (n x m), an array of stacked vectorized images, 
    return a matrix of vectorized patches horizontally patched per image"""
    n_x, m = X.shape
    patch_size = patch_x_size * patch_y_size
    n_patches = int((orig_x_size - patch_x_size)/x_stride) * int((orig_y_size - patch_y_size)/y_stride)
    
    # Build mapping matrix M (patch size * n_patches x n)
    M = np.zeros((patch_size * n_patches, n_x))
    
    # for each patch with a corner at cx, cy
    cursor = 0 
    for px in range(0, orig_x_size - patch_x_size, x_stride):
        for py in range(0, orig_y_size - patch_y_size, y_stride):
            for ix in range(patch_x_size):
                for iy in range(patch_y_size):
                    M[cursor, px * x_stride + ix + (py * y_stride + iy) * orig_x_size] = 1
                    cursor += 1
    # return dot product of M and X
    return np.dot(X, M)

