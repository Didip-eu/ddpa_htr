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
ds_train = pds.PageDataset( from_page_files=imgs_train, polygon_key='coords')
print(ds_train)
ds_val = pds.PageDataset( from_page_files=imgs_val, polygon_key='coords')
print(ds_val)
sys.exit()
aug = tsf.build_tormentor_augmentation_for_crop_training( crop_size=args.img_size, crop_before=False )
ds_train = tormentor.AugmentedDs( ds_train, aug, computation_device='cpu', augment_sample_function=lsg.LineDetectionDataset.augment_with_bboxes )

if 'train' in args.subsets:
    ds_train_cached = lsg.CachedDataset( data_source = ds_train )
    ds_train_cached.serialize( subdir='cached_train', repeat=args.repeat)

# for validation and test, only crops
ds_val = lsg.LineDetectionDataset( imgs_val, lbls_val, min_size=args.img_size, polygon_key='coords')
augCropCenter = tormentor.RandomCropTo.new_size( args.img_size, args.img_size )
augCropLeft = tormentor.RandomCropTo.new_size( args.img_size, args.img_size ).override_distributions( center_x=tormentor.Uniform((0, .6)))
augCropRight = tormentor.RandomCropTo.new_size( args.img_size, args.img_size ).override_distributions( center_x=tormentor.Uniform((.4, 1)))
aug = ( augCropCenter ^ augCropLeft ^ augCropRight ).override_distributions(choice=tormentor.Categorical(probs=(.33, .34, .33)))
ds_val = tormentor.AugmentedDs( ds_val, aug, computation_device='cpu', augment_sample_function=lsg.LineDetectionDataset.augment_with_bboxes )

if 'val' in args.subsets:
    ds_val_cached = lsg.CachedDataset( data_source = ds_val )
    ds_val_cached.serialize( subdir='cached_val', repeat=args.repeat)

ds_test = lsg.LineDetectionDataset( imgs_test, lbls_test, min_size=args.img_size, polygon_key='coords')
ds_test = tormentor.AugmentedDs( ds_test, aug, computation_device='cpu', augment_sample_function=lsg.LineDetectionDataset.augment_with_bboxes )

if 'test' in args.subsets:
    ds_test_cached = lsg.CachedDataset( data_source = ds_test )
    ds_test_cached.serialize( subdir='cached_test', repeat=4)

