import numpy as np
import math
import scipy
import scipy.sparse.linalg

def gray2rgb(img):
    H, W = img.shape
    rgb = np.tile(img.reshape(H, W, -1), (1, 1, 3))
    return rgb

def get_dark_channel(img, win_size=15):
    """
    get the dark channel image from the input hazy image by iterating a 15*15 image patch.
    :param img: [H, W, 3] np array of the input hazy image, pixel value in (0, 1).
    :param win_size: default=15 the window size of the image patch.
    :return:
        dark_channel: [H, W] np array of the dark channel image of the input hazy image.

    """
    H, W, C = img.shape
    pad_size = math.floor(win_size/2)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    padded_r = np.pad(r, pad_size, 'constant', constant_values=np.inf)
    padded_g = np.pad(g, pad_size, 'constant', constant_values=np.inf)
    padded_b = np.pad(b, pad_size, 'constant', constant_values=np.inf)
    padded_img = np.stack((padded_r, padded_g, padded_b), axis=-1)
    dark_channel = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            patch = padded_img[i:i+win_size, j:j+win_size, :]
            dark_channel[i][j] = np.min(patch)
    return dark_channel

def get_atmosphere(img, dark):
    H, W, C = img.shape
    n_pixels = H * W
    n_search_pixels = max(math.floor(n_pixels * 0.001), 1)
    dark_vec = dark.reshape(n_pixels)
    img_vec = img.reshape(n_pixels, -1)
    indices = np.argsort(-dark_vec)
    accumulator = np.zeros((1, 3))
    for i in range(n_search_pixels):
        accumulator += img_vec[indices[i], ...]
    atmosphere = accumulator / n_search_pixels
    return atmosphere

def get_transmission_estimate(img, atmosphere, omega=0.95, win_size=15):
    H, W, C = img.shape
    A = np.tile(atmosphere.reshape(1, 1, 3), (H, W, 1))
    trans_est = 1.0 - omega * get_dark_channel(img / A, win_size)
    return trans_est

def ind2sub(H, W, idx):
    return idx % H, idx // H

def get_laplacian(img):
    H, W, C = img.shape
    img_size = H * W
    epsilon = 0.0000001
    win_rad = 1
    max_num_neigh = (2 * win_rad + 1) ** 2
    ind_mat = np.arange(img_size).reshape((W, H)).T
    indices = np.arange(img_size)
    num_ind = indices.shape[0]
    max_num_vertex = max_num_neigh * max_num_neigh * num_ind

    row_inds = np.zeros((max_num_vertex, 1))
    col_inds = np.zeros((max_num_vertex, 1))
    vals = np.zeros((max_num_vertex, 1))

    len = 0

    for k in range(num_ind):
        print(k)
        ind = indices[k]
        i, j = ind2sub(H, W, ind)
        H_min = max(0, i - win_rad)
        H_max = min(H, i + win_rad)
        W_min = max(0, j - win_rad)
        W_max = min(W, j + win_rad)

        win_inds = ind_mat[H_min: H_max + 1, W_min: W_max + 1]
        win_inds = win_inds.T.reshape(-1)

        num_neigh = win_inds.size

        win_image = img[H_min: H_max + 1, W_min: W_max + 1, :]

        win_image = np.transpose(win_image, (1, 0, 2)).reshape((-1, C))

        win_mean = np.mean(win_image, axis=0, keepdims=True)
        win_var = np.linalg.inv(
            (win_image.T @ win_image / num_neigh)
            - (win_mean.T @ win_mean)
            + (epsilon / num_neigh * np.eye(C))
        )
        win_image = win_image - np.tile(win_mean, (num_neigh, 1))
        win_vals = (1 + win_image @ win_var @ win_image.T) / num_neigh

        # finish computing omega for each image patch
        # start to construct the Laplacian matrix
        sub_len = num_neigh * num_neigh
        win_inds = np.tile(win_inds.reshape((1, -1)), (num_neigh, 1))
        row_inds[len: len + sub_len, ...] = win_inds.reshape((-1, 1))
        win_inds = win_inds.T
        col_inds[len: len + sub_len, ...] = win_inds.reshape((-1, 1))
        vals[len: len + sub_len, ...] = win_vals.T.reshape((-1, 1))
        len = len + sub_len
        print("good")

    A = scipy.sparse.csc_matrix((vals[:len].reshape(-1), (row_inds[:len].reshape(-1), col_inds[:len].reshape(-1))),
                   shape=(img_size, img_size))
    D = scipy.sparse.spdiags(A.sum(1).reshape(-1), [0], img_size, img_size)
    L = D - A
    return L

def get_radiance(img, t, A):
    H, W, C = img.shape
    rep_A = np.tile(A.reshape((1, 1, 3)), (H, W, 1))
    max_t = np.maximum(0.1 * np.ones_like(t), t)
    radiance = (img - rep_A) / max_t.reshape((H, W, -1)) + rep_A
    return radiance