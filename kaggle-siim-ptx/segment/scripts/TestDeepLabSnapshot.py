import sys ; sys.path.insert(0, '..') ; sys.path.insert(0, '../..') 

from peepdom.deeplab import DeepLab

from reproducibility import set_reproducibility

from data.loader import XrayMaskDataset
import pickle

from tqdm import tqdm 
import torch
from torch import optim
from torch import nn
import adabound

import argparse 
import pandas as pd 
import numpy as np 
import glob, os 

from utils.aug import simple_aug, resize_aug, pad_image
from utils.helper import LossTracker, preprocess_input

from torch.utils.data import DataLoader
from functools import partial 

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str)
    parser.add_argument('model_folder', type=str, help="Path to folder containing snapsnot ensemble models.") 
    parser.add_argument('data_dir', type=str, help="Directory to load image data from.") 
    parser.add_argument('mask_dir', type=str, help="Directory to load mask data from.") 
    parser.add_argument('save_file', type=str)
    parser.add_argument('--class-mode', action='store_true')
    parser.add_argument('--inner-fold', type=int, default=0)
    parser.add_argument('--outer-fold', type=int, default=0)
    parser.add_argument('--outer-only', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--labels-df', type=str, default='../../data/train_labels_with_splits.csv')
    parser.add_argument('--imsize-x', type=int, default=384)
    parser.add_argument('--imsize-y', type=int, default=384)
    parser.add_argument('--imratio', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tta', action='store_true', help='Enable test-time augmentation')
    parser.add_argument('--dropout-p', type=float, default=0.2)
    parser.add_argument('--gn', action='store_true')
    parser.add_argument('--output-stride', type=int, default=16)
    parser.add_argument('--verbosity', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=88)
    
    args = parser.parse_args()
    return args 


def main():
    args = parse_args()

    set_reproducibility(args.seed)

    resize_me = resize_aug(imsize_x=args.imsize_x, imsize_y=args.imsize_y)
    pad_func  = partial(pad_image, ratio=args.imratio)

    print ("Testing the PNEUMOTHORAX SEGMENTATION model...")

    torch.cuda.set_device(args.gpu) ; torch.backends.cudnn.benchmark = True 

    if not os.path.exists(os.path.dirname(args.save_file)):
        os.makedirs(os.path.dirname(args.save_file))

    print("Reading labels from {}".format(args.labels_df))

    df = pd.read_csv(args.labels_df)

    if args.outer_only:
        # Test on outer fold
        test_df = df[df['outer'] == args.outer_fold]
    else:
        # Get rid of outer fold test set
        df = df[df['outer'] != args.outer_fold]
        assert np.sum(df['inner{}'.format(args.outer_fold)] == 888) == 0
        test_df = df[df['inner{}'.format(args.outer_fold)] == args.inner_fold]

    print ('TEST: n={}'.format(len(test_df)))

    print("Reading images from directory {}".format(args.data_dir))
    test_images = [os.path.join(args.data_dir, '{}.png'.format(_)) for _ in test_df['sop']]
    test_masks  = [os.path.join(args.mask_dir, '{}.png'.format(_)) for _ in test_df['sop']]
    test_labels = list(test_df['ptx_binary'])
    test_sops = list(test_df['sop'])
    num_classes = 2
    
    # Get models in snapshot ensemble
    snapshots = glob.glob(os.path.join(args.model_folder, '*.pth'))

    num_snapshots = 3
    weights = np.asarray([3.,1.,1.])
    weights = weights / np.sum(weights)
    # Pick best 3 models, then weight based on Kaggle metric: 3, 1, 1
    # This assumes a certain formatting of the checkpoint file name
    # in order to extract the Kaggle metric 
    if args.class_mode:
        def extract_kag(ckpt):
            ckpt = ckpt.split('/')[-1]
            _kag = ckpt.split('_')[4]
            _kag = _kag.split('-')[-1]
            return float(_kag)
    else:
        def extract_kag(ckpt):
            ckpt = ckpt.split('/')[-1]
            _kag = ckpt.split('_')[2]
            _kag = _kag.split('-')[-1]
            return float(_kag)

    snapshot_kags = [extract_kag(_) for _ in snapshots]
    kag_order = np.argsort(snapshot_kags)[::-1][:num_snapshots]
    snapshots = list(np.asarray(snapshots)[kag_order])

    def load_model(ckpt):
        model = DeepLab(args.model, args.output_stride, args.gn, classifier=False)
        model.load_state_dict(torch.load(ckpt))
        model = model.cuda()
        model.eval()
        return model

    # Get models
    print ('Loading checkpoints ...')
    model_list = []
    for ss in snapshots:
        model_list.append(load_model(ss))

    # Set up preprocessing function with model 
    ppi = partial(preprocess_input, model=model_list[0])

    print ('Setting up data loaders ...')

    params = {'batch_size':  1 if args.tta else args.batch_size, 
              'shuffle':     False, 
              'num_workers': args.num_workers}

    test_set = XrayMaskDataset(imgfiles=test_images,
                               maskfiles=test_masks,
                               dicom=False,
                               labels=test_labels,
                               preprocess=ppi, 
                               pad=pad_func,
                               crop=None,
                               resize=resize_me,
                               test_mode=True)
    test_gen = DataLoader(test_set, **params) 

    # Test
    def get_test_predictions(mod):
        with torch.no_grad():
            list_of_pred_dicts = []
            for data in tqdm(test_gen, total=len(test_gen)):
                pred_dict = {}
                if args.tta: 
                    # should be batch size = 1
                    batch, masks, classes = data
                    batch = batch[0]
                    output = mod(batch.cuda())
                    pred_dict['pred_mask'] = torch.softmax(output, dim=1).cpu().numpy()[:,1]
                    pred_dict['gt_mask'] = masks.cpu().numpy().astype('uint8')
                    pred_dict['y_true'] = classes.cpu().numpy()
                else:
                    batch, masks, classes = data
                    output = mod(batch.cuda())
                    output_flipped = mod(torch.flip(batch, dims=(-1,)).cuda())
                    output_flipped = torch.flip(output_flipped, dims=(-1,))
                    pred_dict['pred_mask'] = (torch.softmax(output, dim=1).cpu().numpy()[:,1] + torch.softmax(output_flipped, dim=1).cpu().numpy()[:,1]) / 2.
                    pred_dict['gt_mask'] = masks.cpu().numpy().astype('uint8')
                    pred_dict['y_true'] = classes.cpu().numpy()
                list_of_pred_dicts.append(pred_dict)
        return list_of_pred_dicts

    y_pred_list = []
    for model in tqdm(model_list, total=len(model_list)):
        tmp_y_pred = get_test_predictions(model)
        y_pred_list.append(tmp_y_pred)

    # Need to average predictions across models
    for each_indiv_pred in range(len(y_pred_list[0])):
        indiv_pred = np.zeros_like(y_pred_list[0][each_indiv_pred]['pred_mask'])
        for each_model_pred in range(len(y_pred_list)):
            indiv_pred += weights[each_model_pred]*y_pred_list[each_model_pred][each_indiv_pred]['pred_mask']
        #indiv_pred /= float(len(y_pred_list))
        assert np.min(indiv_pred) >= 0 and np.max(indiv_pred) <= 1
        y_pred_list[0][each_indiv_pred]['pred_mask'] = (indiv_pred * 100).astype('uint8')

    def get_top_X(segmentation, tops=[0,0.5,1.0,2.5,5.0]):
        # Assumes segmentation.shape is (1, H, W)
        assert segmentation.shape[0] == 1
        scores = []
        segmentation = segmentation.reshape(segmentation.shape[0], -1).astype('int8')
        segmentation = -np.sort(-segmentation, axis=1)
        for t in tops:
            size = int(t / 100. * np.prod(segmentation.shape)) if t > 0 else 1
            scores.append(np.mean(segmentation[:,:size]) / 100.)
        return scores

    if args.class_mode:
        # Turn segmentation output into class scores
        tops = [0,0.5,1.0,2.5,5.0]
        class_scores = []
        for i in range(len(y_pred_list[0])):
            class_scores.append(get_top_X(y_pred_list[0][i]['pred_mask'], tops))
        # Make a DataFrame
        class_scores = np.vstack(class_scores)
        class_scores = pd.DataFrame(class_scores)
        class_scores.columns = ['Top{}'.format(t) for t in tops]
        class_scores['y_true'] = [_['y_true'][0] for _ in y_pred_list[0]]
        class_scores['sop'] = test_sops
        class_scores.to_csv(args.save_file, index=False)
    else:

        y_pred_to_pickle = y_pred_list[0]
        y_pred_to_pickle = {test_sops[_] : y_pred_to_pickle[_] for _ in range(len(test_sops))}

        with open(args.save_file, 'wb') as f: 
            pickle.dump(y_pred_to_pickle, f)

if __name__ == '__main__':
    main()



