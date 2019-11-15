import sys ; sys.path.insert(0, 'kaggle-siim-ptx/segment/')

from model.deeplab_jpu import DeepLab

import torch
import torch.nn as nn
import torch.optim as optim

import argparse 
import pydicom
import numpy as np 
import glob, os 
import json

from utils.aug import resize_aug, pad_image
from utils.helper import preprocess_input

from functools import partial 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dicom_folder', type=str)
    #parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args 

def main():
    args = parse_args()

    resize_me = resize_aug(imsize_x=1024, imsize_y=1024)
    pad_func  = partial(pad_image, ratio=1)

    #torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True

    num_classes = 2

    snapshots = ['/opt/kaggle-siim-ptx/checkpoints/deeplab0.pth',
                 '/opt/kaggle-siim-ptx/checkpoints/deeplab1.pth',
                 '/opt/kaggle-siim-ptx/checkpoints/deeplab2.pth']

    def load_model(ckpt):
        model = DeepLab('resnext101_gn_ws', output_stride=16, group_norm=True, center='aspp', jpu=False, use_maxpool=True)
        model.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage))
        #model = model.cuda()
        model.eval()
        return model

    # Get models
    print ('Loading checkpoints ...')
    model_list = []
    for ss in snapshots:
        model_list.append(load_model(ss))

    # Set up preprocessing function with model 
    ppi = partial(preprocess_input, model=model_list[0])

    # Assumes there is 1 DICOM file in the directory 
    # with extension *.dcm
    dicom_file = glob.glob(os.path.join('/io', args.dicom_folder, '*.dcm'))[0]
    dicom = pydicom.dcmread(dicom_file)
    array = dicom.pixel_array
    ORIGINAL_DIMS = array.shape
    # Map to 8-bit image based on DICOM metadata
    try:
        array = np.clip(array, int(dicom.WindowCenter)-int(dicom.WindowWidth)/2, int(dicom.WindowCenter)+int(dicom.WindowWidth)/2)
    except:
        pass
    array = array - np.min(array)
    array = array / np.max(array)
    array = array * 255.
    array = array.astype('uint8')
    #
    try:
        if dicom.PhotometricInterpretation == 'MONOCHROME1':
            array = np.invert(array)
    except:
        pass
    array, pad_list = pad_func(np.expand_dims(array, axis=-1))[...,0]
    array = resize_me(image=array)['image']
    array = ppi(array.astype('float32'))
    array = np.expand_dims(array, axis=0)
    array = np.repeat(array, 3, axis=0)
    array = np.expand_dims(array, axis=0)
    assert array.shape == (1, 3, 1024, 1024)
    array = torch.from_numpy(array).float()
    #array = array.cuda()

    with torch.no_grad():
        output_list = []
        for m in model_list:
            output = m(array)
            output = torch.softmax(output, dim=1).cpu().numpy()[:,1]
            output_list.append(output[0])
            output_flipped = m(torch.flip(array, dims=(-1,)))
            output_flipped = torch.flip(output_flipped, dims=(-1,))
            output_flipped = torch.softmax(output_flipped, dim=1).cpu().numpy()[:,1]
            output_list.append(output_flipped[0])

    outputs = np.asarray(output_list)
    outputs = np.mean(outputs, axis=0)
    outputs = outputs * 255.
    assert outputs.shape == (1024, 1024)

    # Resize back to original shape
    MAX_DIM = np.max(ORIGINAL_DIMS)
    resize_back = resize_aug(MAX_DIM, MAX_DIM)
    outputs = resize_back(image=outputs)['image']
    # Unpad
    outputs = outputs[pad_list[0][0]:(outputs.shape[0]-pad_list[0][1]),pad_list[1][0]:(outputs.shape[1]-pad_list[1][1])]
    assert outputs.shape == ORIGINAL_DIMS

    outputs.astype('uint8').tofile('/io/prob-mask-0.bin')

    json_output = {"protocol_version":"1.0",
                   "parts":[{"label":"segmentation_0",
                             "binary_type":"probability_mask",
                             "binary_data_shape":{"width":1024,
                                                  "height":1024
                                                 }
                            }
                           ]
                  }

    with open('/io/output.json', 'w') as f:
        json.dump(json_output, f)
    

if __name__ == '__main__':
    main()




