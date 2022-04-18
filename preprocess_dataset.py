from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    print(point.shape)
    square = np.sum(point*points, axis=1)
    print(square.shape)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '.mat').replace('images','ground-truth').replace('IMG','GT_IMG')
    points = loadmat(mat_path)['image_info'][0][0][0][0][0].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='./datasets/ShanghaiTech/part_B',
                        help='original data directory')
    parser.add_argument('--data-dir', default='./datasets/ShanghaiTech/part_B',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048
    for phase in ['train_data', 'test_data']:
        sub_dir = os.path.join(args.origin_dir, phase)
        if phase == 'train_data':
            sub_phase_list = ['train', 'val']
            processed_sub_phase = ['processed_train', 'processed_val']
            for proc,sub_phase in zip(processed_sub_phase, sub_phase_list):
                f = os.listdir(os.path.normpath(os.path.join(sub_dir, sub_phase, 'images')))
                sub_save_dir = os.path.normpath(os.path.join(sub_dir, proc))
                for i in f:
                    im_path = os.path.normpath(os.path.join(sub_dir, sub_phase, 'images', i.strip()))
                    name = os.path.basename(im_path)
                    print(im_path)
                    im, points = generate_data(im_path)
                    if sub_phase == 'train':
                        dis = find_dis(points)
                        points = np.concatenate((points, dis), axis=1)
                    im_save_path = os.path.join(sub_save_dir,'images', name)
                    im.save(im_save_path)
                    gd_save_path = im_save_path.replace('jpg', 'npy').replace('images','ground-truth')
                    np.save(gd_save_path, points)
        else:
            sub_phase_list = ['test']
            processed_sub_phase = ['processed_test']
            for proc,sub_phase in zip(processed_sub_phase, sub_phase_list):
                f = os.listdir(os.path.normpath(os.path.join(sub_dir, sub_phase, 'images')))
                sub_save_dir = os.path.normpath(os.path.join(sub_dir, proc))
                for i in f:
                    im_path = os.path.normpath(os.path.join(sub_dir, sub_phase, 'images', i.strip()))
                    name = os.path.basename(im_path)
                    print(im_path)
                    im, points = generate_data(im_path)
                    im_save_path = os.path.join(sub_save_dir,'images', name)
                    im.save(im_save_path)
                    gd_save_path = im_save_path.replace('jpg','npy').replace('images','ground-truth')
                    np.save(gd_save_path, points)
