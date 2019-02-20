"""
- Load probabilistic edge from HED
- Load segmentation probs from deeplab
- Fuse them
"""


import os
import time

import cv2
import numpy as np


DATA_ROOT_DIR = '/home/gpu_user/assia/ws/datasets/kitti'
SEG_DIR = '/mnt/dataX/assia/kitti_deeplab/kitti'
SEQ_L = ['%02d'%d for d in range(11)]
IMG_SUBDIR = 'image_2'
RES_DIR = './res'
NUM_CLASS = 19

for seq in SEQ_L:

    # prepare output dir
    out_dir = os.path.join(RES_DIR, seq)
    for i in range(NUM_CLASS):
        class_dir = os.path.join(out_dir, 'class_%d'%i)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    edge_dir = os.path.join(RES_DIR, seq, 'fuse')
    seg_dir = os.path.join(SEG_DIR, seq, 'prob')

    # let's go
    img_dir = os.path.join(DATA_ROOT_DIR, seq, IMG_SUBDIR)
    for img_root_fn in sorted(os.listdir(img_dir))[:10]:
        print(img_root_fn)

        img_fn  = os.path.join(img_dir, img_root_fn)
        edge_fn = os.path.join(edge_dir, img_root_fn)

        edge = cv2.imread(edge_fn)
        edge = edge.astype(np.float32)/255.0

        for i in range(NUM_CLASS):
            seg_fn  = os.path.join(seg_dir, 'class_%d'%i, img_root_fn)
            seg = cv2.imread(seg_fn)
            seg = seg.astype(np.float32)/255.0

            seg_edge = seg * edge
            seg_edge = (seg_edge*255.0).astype(np.uint8)
            seg_edge_fn = os.path.join(out_dir, 'class_%d'%i, img_root_fn)
            cv2.imwrite(seg_edge_fn, seg_edge)
    
    exit(0)







