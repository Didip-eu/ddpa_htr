#!/usr/bin/env python3
"""
Given a dataset of images+PageXML that has been previously extracted into a page work folder:

1. Split the original set 
2. Out of the train pages, generate a 'Tormented' version and dump the lines
3. Out of the validation pages, generate a plain version and dump the lines

Usage:

```
# generate training and validation sets, with 6 patches out of every source image
PYTHONPATH=. ./bin/ddp_generate_cached_datasets.py -img_paths dataset/*.img.jpg -repeat 6
# generate only validation set
PYTHONPATH=. ./bin/ddp_generate_cached_datasets.py -img_paths dataset/*.img.jpg -repeat 6 -subsets val
```

"""

import tormentor
import math
from pathlib import Path
import sys
import fargv
import random
import tormentor
import matplotlib.pyplot as plt
import torch

sys.path.append( str(Path(__file__).parents[1] ))

from libs import charter_page_ds as pds
from libs import transforms as tsf
from libs.train_utils import split_set


p = {
        'img_paths': set(list(Path("dataset").glob('*.jpg'))),
        'repeat': (1, "Number of patch samples to generate from one image."),
        'subsets': set(['train', 'val']),
        'log_tsv': 1,
        'dummy': 0,
        'img_suffix': '.jpg',
        'lbl_suffix': '.xml',
        'visual_check': (0, "Dry-run: no serialization + visual check of transformed samples."),
}


default_tormentor_dists = {
        'Rotate': tormentor.Uniform((math.radians(-25.0), math.radians(25.0))),
        'Perspective': (tormentor.Uniform((0.85, 1.25)), tormentor.Uniform((.85,1.25))),
        'Wrap': (tormentor.Uniform((0.1, 0.6)), tormentor.Uniform((0.1,0.15))), # quite rough (low-scale), but low intensity
        'Zoom': tormentor.Uniform((1.1,1.6)),
        'Brightness': tormentor.Uniform((-0.25,0.25)),
        'PlasmaBrightness': (tormentor.Uniform((0.1, 0.6)), tormentor.Uniform((0.1,0.5))), # quite rough (low-scale), but low intensity
}


args, _ = fargv.fargv( p )

random.seed(46)
imgs = list([ Path( ip ) for ip in args.img_paths ])

dataset_dirs = list(set( [ str(img.parent) for img in imgs ] ))
if len(dataset_dirs) > 1:
    print('All images should be in the same directory.')
    sys.exit()
dataset_dir = dataset_dirs[0]

imgs_train, imgs_test = split_set( imgs )
imgs_train, imgs_val = split_set( imgs_train )

if args.log_tsv:
    for subset, log_tsv_file in ((imgs_train, 'train_ds.tsv'), (imgs_val, 'val_ds.tsv'), (imgs_test, 'test_ds.tsv')):
        tsv_path = imgs[0].parent.joinpath(log_tsv_file)
        with open( imgs[0].parent.joinpath(log_tsv_file), 'w') as tsv:
            for path in subset:
                tsv.write('{}\t{}\n'.format(path.name, path.name.replace(args.img_suffix, args.lbl_suffix)))
if args.dummy:
    sys.exit()


# for training, Torment at will
ds_train = pds.PageDataset( from_page_files=imgs_train, polygon_key='coords', line_work_folder='dataset/htr_line_dataset/train')
ds_val = pds.PageDataset( from_page_files=imgs_val, polygon_key='coords', line_work_folder='dataset/htr_line_dataset/val')


augWrap = tormentor.RandomWrap.override_distributions(roughness=default_tormentor_dists['Wrap'][0], intensity=default_tormentor_dists['Wrap'][1])
augBrightness = tormentor.RandomBrightness.override_distributions(brightness=default_tormentor_dists['Brightness'])
augPlasmaBrightness = tormentor.RandomPlasmaBrightness.override_distributions(roughness=default_tormentor_dists['PlasmaBrightness'][0], intensity=default_tormentor_dists['PlasmaBrightness'][1])
ds_train.augmentation_class = augPlasmaBrightness #| augWrap



#ds_aug = tormentor.AugmentedDs( ds_train, aug, computation_device='cpu', augment_sample_function=pds.PageDataset.augment_with_bboxes )
    
if args.visual_check:
    plt.close()
    for i in range(10):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(15, 4))
        tsf_sample = ds_train[i]
        ax0.imshow( tsf_sample[0].permute(1,2,0))
        ax1.imshow( torch.sum( tsf_sample[1]['masks'], axis=0) )
        #tsf_sample = ds_aug[i]
        #ax2.imshow( tsf_sample[0].permute(1,2,0))
        #ax3.imshow( torch.sum( tsf_sample[1]['masks'], axis=0) )
        plt.show()

    sys.exit()

if 'train' in args.subsets:
    #ds_train_cached = pds.CachedDataset( data_source = ds_train )
    #ds_train_cached.serialize( subdir='cached_train', repeat=args.repeat)
    ds_train.dump_lines()

if 'val' in args.subsets:
    #ds_val_cached = pds.CachedDataset( data_source = ds_val )
    #ds_val_cached.serialize( subdir='cached_val', repeat=args.repeat)
    ds_val.dump_lines()


