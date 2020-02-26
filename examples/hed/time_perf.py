
import os

import argparse
import time

import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
import Image
import scipy.io


DATA_ROOT_DIR = '/home/gpu_user/assia/ws/datasets/kitti'
SEQ_L = ['%02d'%d for d in range(1)]
IMG_SUBDIR = 'image_2'
RES_DIR = './res'
NEW_W, NEW_H = 1242,375
IMG_NUM = 100

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)

#Visualization
def plot_single_scale(scale_lst, size, out_fn):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(1,5,i+1)
        plt.imshow(1-scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    print(out_fn)
    plt.savefig(out_fn)
    plt.close()

def assemble_multiscale(img_fn_l):
    """
    Args:
        img_fn_l: list of 5 images
    """
    h,w = img_fn_l[0].shape[:2]
    
    for i, img in enumerate(img_fn_l):
        print(np.max(img))
        img_fn_l[i] = 1 - img
    toto = (np.ones((img_fn_l[0].shape))*255).astype(np.uint8)
    line1 = np.hstack((img_fn_l[0], img_fn_l[1], img_fn_l[2]))
    line2 = np.hstack((img_fn_l[3], img_fn_l[4], toto))
    
    out = np.vstack((line1, line2))
    out[:,w:w+1] = 0
    out[:,2*w:2*w+1] = 0
    out[h:h+1,:] = 0
    return out

# load net
model_root = './'
net = caffe.Net(model_root+'deploy.prototxt', model_root+'hed_pretrained_bsds.caffemodel', caffe.TEST)

global_start_time = time.time()
duration = 0.0
for seq in SEQ_L:

    # let's go
    img_dir = os.path.join(DATA_ROOT_DIR, seq, IMG_SUBDIR)
    for img_root_fn in sorted(os.listdir(img_dir))[:IMG_NUM]:
        img_fn = os.path.join(img_dir, img_root_fn)
        in_ = cv2.imread(img_fn)
        in_ = cv2.resize(in_, (NEW_W, NEW_H), interpolation=cv2.INTER_AREA)
        in_ = in_.astype(np.float32)
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
    
        # shape for input (data blob is N x C x H x W), set data
        in_ = in_.transpose((2,0,1))
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        start_time = time.time()
        net.forward()
        fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
        duration += time.time() - start_time
    
    print('seq: %s - %s - %d:%02d - %ds - %ds/img'
            %(seq, img_root_fn, duration/60, duration%60, duration,
                1.0*duration/IMG_NUM))

