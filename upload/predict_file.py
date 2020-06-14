import os
import cv2
import h5py
import numpy as np
from keras.engine.saving import model_from_json
from scipy.io import loadmat
from tqdm import tqdm
import os
import cv2
import glob
import h5py
import scipy
import tensorflow as tf
from scipy import integrate
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.io import loadmat, savemat

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gen_var_from_paths(paths, stride=1, unit_len=16):
    vars = []
    format_suffix = paths[0].split('.')[-1]
    if format_suffix == 'h5':
        for ph in paths:
            dm = h5py.File(ph, 'r')['density'].value.astype(np.float32)
            if unit_len:
                dm = fix_singular_shape(dm, unit_len=unit_len)
            dm = smallize_density_map(dm, stride=stride)
            vars.append(np.expand_dims(dm, axis=-1))
    elif format_suffix == 'jpg' or format_suffix == 'JPG':
        for ph in paths:
            raw = cv2.cvtColor(cv2.imread(ph), cv2.COLOR_BGR2RGB).astype(np.float32)
            if unit_len:
                raw = fix_singular_shape(raw, unit_len=unit_len)
            vars.append(raw)
        # vars = norm_by_imagenet(vars)
    else:
        print('Format suffix is wrong.')
    return np.array(vars)


def flip_horizontally(x, y):
    to_flip = np.random.randint(0, 2)
    if to_flip:
        x, y = cv2.flip(x, 1), np.expand_dims(cv2.flip(np.squeeze(y), 1), axis=-1)
        # Suppose shape of y is (123, 456, 1), after cv2.flip, shape of y would turn into (123, 456).
    return x, y


def smallize_density_map(density_map, stride=1):
    if stride > 1:
        density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int) // stride).tolist(),
                                      dtype=np.float32)
        for r in range(density_map_stride.shape[0]):
            for c in range(density_map_stride.shape[1]):
                density_map_stride[r, c] = np.sum(
                    density_map[r * stride:(r + 1) * stride, c * stride:(c + 1) * stride])
    else:
        density_map_stride = density_map
    return density_map_stride


def fix_singular_shape(img, unit_len=16):
    """
    Some network like w-net has both N maxpooling layers and concatenate layers,
    so if no fix for their shape as integeral times of 2 ** N, the shape will go into conflict.
    """
    hei_dst, wid_dst = img.shape[0] + (unit_len - img.shape[0] % unit_len), img.shape[1] + (
            unit_len - img.shape[1] % unit_len)
    if len(img.shape) == 3:
        img = cv2.resize(img, (wid_dst, hei_dst), interpolation=cv2.INTER_LANCZOS4)
    elif len(img.shape) == 2:
        GT = int(round(np.sum(img)))
        img = cv2.resize(img, (wid_dst, hei_dst), interpolation=cv2.INTER_LANCZOS4)
        img = img / (np.sum(img) / GT)
    return img


def norm_by_imagenet(img):
    if len(img.shape) == 3:
        img = img / 255.0
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
        return img
    elif len(img.shape) == 4 or len(img.shape) == 1:
        # In SHA, shape of images varies, so the array.shape is (N, ), that's the '== 1' case.
        imgs = []
        for im in img:
            im = im / 255.0
            im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
            im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
            im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
            imgs.append(im)
        return np.array(imgs)
    else:
        print('Wrong shape of the input.')
        return None


# Analysis on results
def test(test_img_path, test_dm_path):
    test_x = gen_var_from_paths(test_img_path[:], unit_len=None)
    test_y = gen_var_from_paths(test_dm_path[:], stride=8, unit_len=None)
    test_x = norm_by_imagenet(test_x)

    dataset = 'A'
    net = 'CSRNet'

    dis_idx = 16 if dataset == 'B' else 0
    weights_dir_neo = 'D:/Crowd Web/crowd_web/upload/weights_A_MSE_bestMAE294.332_Sun-Jul-14'
    model = model_from_json(open('models/{}.json'.format(net), 'r').read())
    model.load_weights(os.path.join(weights_dir_neo, '{}_best.hdf5'.format(net)))
    ct_preds = []
    ct_gts = []

    for i in range(len(test_x[:])):
        if i % 100 == 0:
            print('{}/{}'.format(i, len(test_x)))
        i += 0
        test_x_display = np.squeeze(test_x[i])
        test_y_display = np.squeeze(test_y[i])
        path_test_display = test_img_path[i]
        pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
        ct_pred = np.sum(pred)
        ct_gt = round(np.sum(test_y_display))
        ct_preds.append(ct_pred)
        ct_gts.append(ct_gt)

    plt.plot(ct_preds, 'r>')
    plt.plot(ct_gts, 'b+')
    plt.legend(['ct_preds', 'ct_gts'])
    plt.title('Pred vs GT')
    plt.show()
    error = np.array(ct_preds) - np.array(ct_gts)
    plt.plot(error)
    plt.title('Pred - GT, mean = {}, MAE={}'.format(
        str(round(np.mean(error), 3)),
        str(round(np.mean(np.abs(error)), 3))
    ))
    plt.show()
    idx_max_error = np.argsort(np.abs(error))[::-1]

    # Show the 5 worst samples
    for worst_idx in idx_max_error[:5].tolist() + [dis_idx]:
        test_x_display = np.squeeze(test_x[worst_idx])
        test_y_display = np.squeeze(test_y[worst_idx])
        path_test_display = test_img_path[worst_idx]
        pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
        fg, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
        ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_test_display), cv2.COLOR_BGR2RGB))
        ax_x_ori.set_title('Original Image')
        ax_y.imshow(test_y_display, cmap=plt.cm.jet)
        ax_y.set_title('Ground_truth: ' + str(np.sum(test_y_display)))
        ax_pred.imshow(pred, cmap=plt.cm.jet)
        ax_pred.set_title('Prediction: ' + str(np.sum(pred)))
        plt.show()


# Analysis on results
def predict(test_img_path):
    from keras import backend as K
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

    print('entered to predict func')
    test_x = gen_var_from_paths(test_img_path[:], unit_len=None)
    test_x = norm_by_imagenet(test_x)
    print(len(test_x))

    dataset = 'A'
    net = 'CSRNet'

    dis_idx = 16 if dataset == 'B' else 0
    weights_dir_neo = 'upload/weights_A_MSE_bestMAE294.332_Sun-Jul-14'
    model = model_from_json(open('upload/models/{}.json'.format(net), 'r').read())
    model.load_weights(os.path.join(weights_dir_neo, '{}_best.hdf5'.format(net)))
    print('model is loaded')
    ct_preds = []

    for i in range(len(test_x[:])):
        if i % 100 == 0:
            print('{}/{}'.format(i, len(test_x)))
        i += 0
        test_x_display = np.squeeze(test_x[i])
        # path_test_display = test_img_path[i]
        pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
        ct_pred = np.sum(pred)
        ct_preds.append(ct_pred)

    # Show the 5 worst samples
    test_x_display = np.squeeze(test_x[0])
    # path_test_display = test_img_path[0]
    print('before predict ')
    # pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
    print('after predict')
    print(
        'before predict crash ********************************************************************************************************************************************************************')

    fg, (ax_pred) = plt.subplots(1, 1, figsize=(6, 4))
    # ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_test_display), cv2.COLOR_BGR2RGB))
    # ax_x_ori.set_title('Original Image')

    ax_pred.imshow(pred, cmap=plt.cm.jet, interpolation='none')

    ax_pred.set_title('Prediction: ' + str(np.sum(pred)))
    # plt.show()

    fg.savefig('media/DM.png')
    plt.close(fg)
    print(
        'after predict crash ********************************************************************************************************************************************************************')

    # test_x = gen_var_from_paths(test_img_path[:], unit_len=None)
    # test_x = norm_by_imagenet(test_x)
    #
    # dataset = 'A'
    # net = 'CSRNet'
    #
    # dis_idx = 16 if dataset == 'B' else 0
    # weights_dir_neo = 'D:/Crowd Web/crowd_web/upload/weights_A_MSE_bestMAE294.332_Sun-Jul-14'
    # model = model_from_json(open('upload/models/{}.json'.format(net), 'r').read())
    # model.load_weights(os.path.join(weights_dir_neo, '{}_best.hdf5'.format(net)))
    # ct_preds = []
    #
    # for i in range(len(test_x[:])):
    #     if i % 100 == 0:
    #         print('{}/{}'.format(i, len(test_x)))
    #     i += 0
    #     test_x_display = np.squeeze(test_x[i])
    #     path_test_display = test_img_path[i]
    #     pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
    #     ct_pred = np.sum(pred)
    #     ct_preds.append(ct_pred)
    #
    # # Show the 5 worst samples
    # test_x_display = np.squeeze(test_x[0])
    # path_test_display = test_img_path[0]
    # pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
    #
    # fg, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
    # ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_test_display), cv2.COLOR_BGR2RGB))
    # ax_x_ori.set_title('Original Image')
    # ax_pred.imshow(pred, cmap=plt.cm.jet)
    # ax_pred.set_title('Prediction: ' + str(np.sum(pred)))
    # #plt.show()


def gen_density_map_gaussian(im, points, sigma=4):
    """
    func: generate the density map
    """
    density_map = np.zeros(im.shape[:2], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map
    if sigma == 4:
        # Adaptive sigma in CSRNet.
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances, _ = tree.query(points, k=4)
    for idx_p, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        gaussian_radius = sigma * 2 - 1
        if sigma == 4:
            # Adaptive sigma in CSRNet.
            sigma = max(int(np.sum(distances[idx_p][1:4]) * 0.1), 1)
            gaussian_radius = sigma * 3
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(int(gaussian_radius * 2 + 1), sigma),
            cv2.getGaussianKernel(int(gaussian_radius * 2 + 1), sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        gaussian_map = gaussian_map[y_up:y_down, x_left:x_right]
        if np.sum(gaussian_map):
            gaussian_map = gaussian_map / np.sum(gaussian_map)
        density_map[
        max(0, p[0] - gaussian_radius):min(h, p[0] + gaussian_radius + 1),
        max(0, p[1] - gaussian_radius):min(w, p[1] + gaussian_radius + 1)
        ] += gaussian_map
    density_map = density_map / (np.sum(density_map / num_gt))
    return density_map


def calculatedesnity(new_img_paths, new_gt_paths):
    print("Calculate density maps:")
    for img_path, gt_path in tqdm(zip(new_img_paths, new_gt_paths)):

        # Load .mat files
        pts = loadmat(gt_path)
        img = cv2.imread(img_path)

        sigma = 3
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = pts["annPoints"]

        for i in range(len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        DM = gen_density_map_gaussian(k, gt, sigma=sigma)
        fg, (ax_pred) = plt.subplots(1, 1, figsize=(6, 4))
        # ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_test_display), cv2.COLOR_BGR2RGB))
        # ax_x_ori.set_title('Original Image')
        ax_pred.imshow(DM, cmap=plt.cm.jet, interpolation='none')
        ax_pred.set_title('Prediction: ' + str(np.sum(DM)))
        # plt.show()
        print(
            'before density crash ********************************************************************************************************************************************************************')
        fg.savefig('media/GT.png')
        plt.close(fg)
        print(
            'after density crash ********************************************************************************************************************************************************************')

# calculatedesnity(['C:/Users/heba/Desktop/CrowdCountingFinal/Data/haramData/images/newImage01.jpg'],
#                  ['C:/Users/heba/Desktop/CrowdCountingFinal/Data/haramData/ground-truth/newImage01.mat'])
#
# test(['C:/Users/heba/Desktop/CrowdCountingFinal/Data/haramData/images/newImage01.jpg'],
#      ['C:/Users/heba/Desktop/CrowdCountingFinal/Data/haramData/ground-truth/newImage01.h5'])

# predict(['C:/Users/heba/Desktop/CrowdCountingFinal/Data/haramData/images/newImage01.jpg'])
