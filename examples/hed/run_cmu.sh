#!/bin/sh

MACHINE=1
if [ "$MACHINE" -eq 0 ]; then # my baby
    WS_DIR=/home/abenbihi/ws/
    CAFFE_PYTHON_DIR="$WS_DIR"tf/CASENet_Codes/caffe/python/
elif [ "$MACHINE" -eq 1 ]; then # docker
    WS_DIR=/home/ws/
    CAFFE_PYTHON_DIR=/opt/caffe/python/
fi

if [ $# -eq 0 ]; then
    echo "Usage"
    echo "1. slice_id"
    echo "2. cam_id"
    echo "3. survey_id"
    exit 0
fi

if [ $# -ne 3 ]; then
    echo "Bad number of arguments"
    echo "1. slice_id"
    echo "2. cam_id"
    echo "3. survey_id"
    exit 1
fi

slice_id="$1"
cam_id="$2"
survey_id="$3"

IMAGE_DIR="$WS_DIR"datasets/cmu/
res_dir=res/cmu/slice"$slice_id"
if [ -d "$res_dir" ]; then
    mkdir -p "$res_dir"
fi


python -m run_cmu \
    --pycaffe_folder ../../python/ \
    --prototxt deploy.prototxt \
    --model hed_pretrained_bsds.caffemodel \
    --slice_id "$slice_id" \
    --cam_id "$cam_id" \
    --survey_id "$survey_id" \
    --img_dir "$IMG_DIR" \
    --res_dir 

