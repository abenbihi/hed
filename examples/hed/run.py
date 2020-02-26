
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
RES_DIR = './res'

SEQ_L = ['%02d'%d for d in range(2,3)]
#SEQ_L = ['%02d'%d for d in range(11)]
# SEQ_L [ '04' ] # to just on run '04'
IMG_SUBDIR = 'image_2'
NEW_W, NEW_H = 1242,375

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

for seq in SEQ_L:

    # prepare output dir
    out_dir = os.path.join(RES_DIR, seq)
    fuse_dir = os.path.join(out_dir, 'fuse')
    if not os.path.exists(fuse_dir):
        os.makedirs(fuse_dir)
    #scale_dir = {}
    #for i in range(1,6):
    #    scale_dir[i] = os.path.join(out_dir, 'scale_%d'%i)
    #    if not os.path.exists(scale_dir[i]):
    #        os.makedirs(scale_dir[i])

    # let's go
    #img_dir = os.path.join(DATA_ROOT_DIR, seq)
    img_dir = "%s/%s/image_2"%(DATA_ROOT_DIR, seq)
    for img_root_fn in sorted(os.listdir(img_dir)):
        fuse_fn = os.path.join(fuse_dir, img_root_fn)
        #if os.path.exists(fuse_fn):
        #    continue
        duration = time.time() - global_start_time
        #print('seq: %s - %s - %d:%02d'%(seq, img_root_fn, duration/60, duration%60))


        img_fn = os.path.join(img_dir, img_root_fn)
        #print(img_fn)
        #im = Image.open(img_fn)
        #in_ = np.array(im, dtype=np.float32)
        #in_ = in_[:,:,::-1]
        in_ = cv2.imread(img_fn)
        in_ = cv2.resize(in_, (NEW_W, NEW_H), interpolation=cv2.INTER_AREA)
        in_ = in_.astype(np.float32)
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
    

        # shape for input (data blob is N x C x H x W), set data
        in_ = in_.transpose((2,0,1))
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        #out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
        #out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
        #out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
        #out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
        #out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
        fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    
        # save edge prob
        #print(fuse.shape)
        #print(fuse.dtype)
        #np.savetxt(fuse_fn, fuse)

        ## save edge img
        #fuse = (255*(fuse)).astype(np.uint8)
        #fuse_fn = os.path.join(fuse_dir, img_root_fn)
        #cv2.imwrite(fuse_fn, fuse)

        ## save multi scale edge prob
        #scale_lst = [out1, out2, out3, out4, out5]
        #for i, out in enumerate(scale_lst):
        #    #out_fn = os.path.join(scale_dir[i+1], img_root_fn.split(".")[0] +'.txt')
        #    #np.savetxt(out_fn, (255*out).astype(np.uint8))
        #    out_fn = os.path.join(scale_dir[i+1], img_root_fn)
        #    cv2.imwrite(out_fn, (255*out).astype(np.uint8))
        
        
        
        #fuse = (1-fuse)
        #print(fuse)
        #print(np.max(fuse))
        #cv2.imshow('fuse', fuse)
        #cv2.imwrite('fuse.png', (255*fuse).astype(np.uint8))
        #cv2.waitKey(0)
        #
        #multi_scale = assemble_multiscale(scale_lst)
        #cv2.imshow('multi_scale', multi_scale)
        #cv2.imwrite('multi_scale.png', (255*multi_scale).astype(np.uint8))
        #cv2.waitKey(0)

    duration = time.time() - global_start_time
    print('seq: %s - %d:%02d'%(seq, duration/60, duration%60))
