"""Run hed on cmu-seaons."""
import os
import argparse
import sys # bad bad bad 
import time

import cv2
import numpy as np
import scipy.misc
import Image
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm

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

def main(args):
    global_start_time = time.time()
    assert(os.path.exists(args.deploy_prototxt_file))
    assert(os.path.exists(args.model))
    
    # caffe setup
    if os.path.exists(args.pycaffe_folder)): # ../../python/
        sys.path.insert(0, args.pycaffe_folder)
    import caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    net = caffe.Net(args.protoxt, args.model, caffe.TEST)
    
    # prepare output dir
    fuse_dir = "%s/fuse/"%args.res_dir
    if not os.path.exists(fuse_dir):
        os.makedirs(fuse_dir)
    
    # set input path
    if args.survey_id == -1:
        meta_fn = "meta/cmu/surveys/%d/c%d_db.txt"%(args.slice_id, args.cam_id)
    else:
        meta_fn = "meta/cmu/surveys/%d/c%d_%d.txt"%(args.slice_id, args.cam_id,
                args.survey_id)
    meta = np.loadtxt(meta_fn, dtype=str)
    img_fn_l = ["%s/%s"%(args.img_dir, l) for l in meta[:,0]]
    root_fn_l = ["%s.png"%(l.split("/")[-1]).split(".")[0] for l in meta[:,0]]


    for idx, img_fn in enumerate(img_fn_l):
        out_fn = "%s/%s.png"%(fuse_dir, root_fn_l[idx])
        if os.path.exists(out_fn): # this img is already done
            continue
        duration = time.time() - global_start_time
        if idx%20 == 0:
            print('%d/%d\t%d:%02d'%(idx, len(root_fn_l), duration/60, duration%60))
    
        in_ = cv2.imread(img_fn)
        in_ = in_.astype(np.float32)
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
    
        # shape for input (data blob is N x C x H x W), set data
        in_ = in_.transpose((2,0,1))
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    
        # save edge img
        fuse = (255*(fuse)).astype(np.uint8)
        fuse_fn = os.path.join(fuse_dir, img_root_fn)
        cv2.imwrite("%s/%s.png"%(fuse_dir, root_fn_l[idx]), fuse)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototxt" type=str,required=True)
    parser.add_argument('--model', type=str, help="path to the caffemodel")
    parser.add_argument('--pycaffe_folder', type=str, default='../../code/python',
                        help="pycaffe folder that contains the caffe/_caffe.so file")
    parser.add_argument('--gpu', type=int, default=0,
                        help="use which gpu device (default=0)")
    
    parser.add_argument('--slice_id', type=int, default=22)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--survey_id', type=int, default=0)
    
    parser.add_argument('--img_dir', type=str, default='', help="image dir")
    parser.add_argument('--res_dir', type=str, default='.', help="folder to store the test results")
