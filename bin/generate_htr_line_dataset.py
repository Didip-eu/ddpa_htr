#!/usr/bin/env python3
"""
Given a dataset of images+PageXML that has been previously extracted into a page work folder,
generate a 'Tormented' version and dump the lines

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

from libs import charter_htr_datasets as pds
from libs import transforms as tsf
from libs.train_utils import split_set


p = {
        'img_paths': [ set([]), 'Image paths'],
        'repeat': [1, "Number of samples to generate from one image."],
        'log_tsv': 0,
        'dummy': 0,
        'img_suffix': '.jpg',
        'lbl_suffix': '.xml',
        'visual_check': [0, "Dry-run: no serialization + visual check of transformed samples."],
        'line_ds_path': ['', "Where lines are to be serialized."]
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

if args.log_tsv:
    log_tsv_file = 'ds.tsv'
    tsv_path = imgs[0].parent.joinpath(log_tsv_file)
    with open( imgs[0].parent.joinpath(log_tsv_file), 'w') as tsv:
        for path in imgs:
            tsv.write('{}\t{}\n'.format(path.name, path.name.replace(args.img_suffix, args.lbl_suffix)))
if args.dummy:
    sys.exit()


# for training, Torment at will
ds = pds.PageDataset( from_page_files=imgs, device='cuda' )


augWrap = tormentor.RandomWrap.override_distributions(roughness=default_tormentor_dists['Wrap'][0], intensity=default_tormentor_dists['Wrap'][1])
augBrightness = tormentor.RandomBrightness.override_distributions(brightness=default_tormentor_dists['Brightness'])
augPlasmaBrightness = tormentor.RandomPlasmaBrightness.override_distributions(roughness=default_tormentor_dists['PlasmaBrightness'][0], intensity=default_tormentor_dists['PlasmaBrightness'][1])
augGaussianAdditiveNoise = tormentor.RandomGaussianAdditiveNoise



# Note: this hides the dump_line() method
# ds_aug = tormentor.AugmentedDs( ds, augGaussianAdditiveNoise|augPlasmaBrightness, computation_device='cuda', augment_sample_function=pds.PageDataset.augment_with_bboxes )
    
if args.visual_check:
    plt.close()
    for i in range(len(ds)):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(15, 4))
        tsf_sample = ds[i]
        ax0.imshow( tsf_sample[0].permute(1,2,0))
        ax1.imshow( torch.sum( tsf_sample[1]['masks'], axis=0) )
        print('\n'.join( tsf_sample[1]['texts']))
        tsf_sample = ds_aug[i]
        ax2.imshow( tsf_sample[0].permute(1,2,0))
        ax3.imshow( torch.sum( tsf_sample[1]['masks'], axis=0) )
        plt.show()

    sys.exit()

if args.line_ds_path:# and Path( args.line_ds_path).exists():
    ds.dump_lines( args.line_ds_path, iteration=0)
    ds.augmentation_class = augGaussianAdditiveNoise|augPlasmaBrightness 
    for rp in range(1,args.repeat+1):
        ds.dump_lines( args.line_ds_path, iteration=rp )


