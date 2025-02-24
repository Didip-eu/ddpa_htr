#!/usr/bin/env python3

"""
Goal: a training/validation dataset + charter test dataset

Training/validation:

+ ~10% Nuremberg letterbooks: ? sample samples with regularized spelling, previously filtered to eliminate strike-throughs
+ ~37.5% Koenigsfelden
+ ~22.5% Monasterium/Teklia
+ ~30% fsdb_aligned

with 84% training, 8% validation, 8% test

Training sample counts:

| Dataset           | Actual     | Combined set   | 
| :---              |       ---: |          ---:  |
Nuremberg           |     26130  |          3760  | 11258/3
Koenigsfelden       |     26167  |         14050  | 37.52/30 * 11258
Monasterium/Teklia  |      8423  |          8423  | Fixed
FSDB_aligned        |     11165  |         11165  | Fixed

Recipe:

Create dataset folder with
+ all fsdb and teklia samples
+ a random selection of 14,000 Koenigsfelden samples
+ a filtered, random selection of 3750 Nuremberg samples
"""

from pathlib import Path
import random
import os
import sys

nuremberg_folder = Path('../nuremberg_matrix')
koenigsfelden_folder = Path('../koenigsfelden_matrix')
fsdb_aligned_folder = Path('../fsdb_aligned_matrix')
MonasteriumTeklia_folder = Path('../MonasteriumTeklia_matrix')

koenigsfelden_count = 14050
nuremberg_count=3760

def purge(folder: str, include_suffix:list=['.npy', '.txt', '.jpg', '.png', '.gz', '.tsv']) -> int:
    cnt = 0
    for item in [ f for f in Path( folder ).iterdir() if not f.is_dir() and f.suffix in include_suffix ]:
        item.unlink()
        cnt += 1
    return cnt


print("Selecting {} samples out of {}".format(koenigsfelden_count, koenigsfelden_folder ))

random.seed( 7 )
koenigsfelden_samples = random.sample( [ f.with_suffix('') for f in koenigsfelden_folder.glob('*.png') ], koenigsfelden_count)
print( len(koenigsfelden_samples))

print("Filtering Nuremberg items...")
filtered_nuremberg_items = []
for gt_file in nuremberg_folder.glob('*.gt.txt'):
    with open(gt_file, 'r') as gt_in:
        if '%' in gt_in.read():
            continue
        filtered_nuremberg_items.append( gt_file )
print("Found {} files.".format(len(filtered_nuremberg_items)))


print("Selecting {} samples out of (filtered) {}".format(nuremberg_count, nuremberg_folder ))
nuremberg_samples = random.sample( [ f.with_suffix('') for f in filtered_nuremberg_items ], nuremberg_count)
print(len(nuremberg_samples))


if not Path('no_links').exists():
    
    purge('.')

    print("Creating hard links to files in {}".format( fsdb_aligned_folder ))
    for file in fsdb_aligned_folder.glob('*'):
        if file.suffix in ['.tsv', '.md', '.py'] or file.is_dir():
            continue
        os.link( file, file.name)

    print("Creating hard links to files in {}".format( MonasteriumTeklia_folder ))
    for file in MonasteriumTeklia_folder.glob('*'):
        if file.suffix in ['.tsv', '.md', '.py'] or file.is_dir():
            continue
        os.link(file, file.name)

    print("Creating hard links to files in {}".format( koenigsfelden_folder ))
    selected_files = [ [ filepath_prefix.with_suffix(sfx) for sfx in ('.gt.txt','.channel.npy.gz', '.bool.npy.gz', '.png') ] for filepath_prefix in koenigsfelden_samples ] 
    for triplet in selected_files:
        for fpath in triplet:
            os.link( fpath, fpath.name)


    print("Creating hard links to files in {}".format( nuremberg_folder ))
    selected_files = [ [ filepath_prefix.with_suffix(sfx) for sfx in ('.gt.txt','.channel.npy.gz', '.bool.npy.gz','.png') ] for filepath_prefix in nuremberg_samples ] 
    for triplet in selected_files:
        for fpath in triplet:
            os.link( fpath, fpath.name)



