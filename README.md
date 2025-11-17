# HTR app

Prototype of a Kraken/VGSL-based, HTR app. The current draft contains:

- a model builder/wrapper `model_htr.py`, that provides the interfaces needed for training and inference.
- a training script `ddp_htr_train.py`, that can use any pre-processed set of line image+transcription samples, preferably stored on-disk (see below).
- a high-level script `ddp_htr_inference.py` that runs inference on FSDB images, each provided with an existing JSON segmentation file, whose default path is ``<input_image_stem>.lines.pred.json``.
- a shell-script wrapper, that manages the dependencies for the HTR task, if ever needed: it relies on a Makefile to generate the segmentation meta-data for the charter, if they are not already present (with `ddpa_lines`) and calls then `ddp_htr_inference.py`.

TODO: 

+ decoding options?
+ integrate PyLemmatizer in the training/inference scripts
+ proper testing (the existing testing modules are obsoletes)
+ ...

## Installing

### Code

```bash
git clone git@github.com:Didip-eu/ddpa_htr.git
cd ddpa_htr
pip install -r requirements.txt
```


### Data

A training set with pre-compiled line images (RGB) and transcriptions can be downloaded from [this location](https://drive.google.com/uc?id=1zhLi1FWCyCdMF3v6HiXxYgbMOTIw2StZ)

Alternatively, run the following commands:

```bash
pip install gdown
cd ./data
gdown https://drive.google.com/uc?id=1zhLi1FWCyCdMF3v6HiXxYgbMOTIw2StZ
unzip MonasteriumTeklia_htr_precompiled.zip
```



## How to use


### 1. Precompiling the dataset pages

The `libs/charter_htr_datasets.py` module defines two classes

+ `PageDataset` handles data at the page level, for 
   - downloading specific HTR datasets
   - compiling and augmenting regions out of pages (images + XML or JSON metadata)
   - extracting and serializing line samples 
+ `HTRLineDataset` uses the resulting line samples for training, with options for masking and line-wide transforms

#### 1.1. Obtaining lines out of pages and regions: the `PageDataset` class

For the sake of clarity, we illustrate the workflow by skipping the downloading stage and assume that the relative location `dataset/page_ds` already contains charter images and their PageXML metadata:

```
1976 dataset/page_ds/NA-CG-L_14300811_206_r.jpg
  40 dataset/page_ds/NA-CG-L_14300811_206_r.xml
1604 dataset/page_ds/NA-CG-L_14310725_216_r.jpg
  16 dataset/page_ds/NA-CG-L_14310725_216_r.xml
  ...
```

We can compile lines out of an explicit list of charter image files through the following steps:

1. Extract text regions:
   
   ```python
   ds=PageDataset( from_page_files=Path('./dataset/page_ds').glob('*216_*r.jpg'))
   2025-11-16 11:49:51,167 - build_page_region_data: Building region data items (this might take a while).
   100%|===================================================================| 4/4 [00:07<00:00,  1.95s/it]
   2025-11-16 11:49:58,963 - __init__:
                   Root path:	data
                   Archive path:	data/-
                   Archive root folder:	data/-
                   Page_work folder:	dataset/page_ds
                   Data points:	5
   ```
   
   Note that the number of data points (i.e. 5 regions) typically exceeds the number of charter images (4 in our example). At this point, the **page work folder** (in addition the pre-existing charter-wide data) contains one image crop and one JSON label file for each region:
   
   ```bash
     140 dataset/page_ds/NA-ACK_14611216_01686_r-r1.json
   19816 dataset/page_ds/NA-ACK_14611216_01686_r-r1.png
       4 dataset/page_ds/NA-ACK_14611216_01686_r-region_1573576805990_657.json
     380 dataset/page_ds/NA-ACK_14611216_01686_r-region_1573576805990_657.png
     120 dataset/page_ds/NA_ACK_338_13500216_r-r1.json
    7064 dataset/page_ds/NA_ACK_338_13500216_r-r1.png
     120 dataset/page_ds/NA_ACK_339_13500216_r-r1.json
    6324 dataset/page_ds/NA_ACK_339_13500216_r-r1.png
      80 dataset/page_ds/NA-CG-L_14330730_216_r-r1.json
    5632 dataset/page_ds/NA-CG-L_14330730_216_r-r1.png
   ```
   
2. Serialize the lines:
   
   ```python
   ds.dump_lines('dataset/htr_line_dataset', overwrite_existing=True)
   100%|===================================================================| 5/5 [00:15<00:00,  3.05s/it]
   ```

   The resulting **line work folder** stores:

   + the line images  (`*.png`)
   + the transcription ground truth files (`*.gt.txt`)
   + binary masks generated from each line polygon boundaries (`*.bool.npy.gz`), to be used at loading time with a masking function of choice.

   ```bash
     4 dataset/htr_line_dataset/445m-r4-0.bool.npy.gz
     4 dataset/htr_line_dataset/445m-r4-0.gt.txt
   304 dataset/htr_line_dataset/445m-r4-0.png
     4 dataset/htr_line_dataset/445m-r4-10.bool.npy.gz
     4 dataset/htr_line_dataset/445m-r4-10.gt.txt
   152 dataset/htr_line_dataset/445m-r4-10.png
   ... 
   ```


Alternatively, to compile lines out of all charters contained in the page work folder, use the `from_page_folder` option:
 
```python
PageDataset( from_page_folder=Path('./dataset/page_ds'), limit=3).dump_lines('dataset/htr_line_dataset', overwrite_existing=True)
2025-11-16 12:14:28,695 - build_page_region_data: Building region data items (this might take a while).
100%|======================================================================| 3/424 [00:01<02:35,  2.71it/s]
2025-11-16 12:14:29,804 - __init__:
                Root path:	data
                Archive path:	data/-
                Archive root folder:	data/-
                Page_work folder:	dataset/page_ds
                Data points:	3

100%|======================================================================| 3/3 [00:02<00:00,  1.16it/s]
2025-11-16 12:14:32,406 - dump_lines: Compiled 74 lines
```

The script `bin/ddp_generate_htr_line_dataset.py` shows how to compile lines out of Tormentor-augmented regions.


#### 1.2. Packing up line samples for training: the `HTRLineDataset` class

Use the `HTRLineDataset`class. It assumes the set has been split before. The respective subsets can be then passsed

+ as a list of image files:

  ```
  ds=HTRLineDataset( from_line_files=Path('./dataset/htr_line_ds').glob('*.png') )
  ```

+ as a TSV file storing a list of samples. Eg.

  ```python
  # create and store as TSV
  HTRLineDataset( from_line_files=Path('./dataset/htr_line_ds').glob('*.png'), to_tsv_file='train_ds.tsv' )
  # instantiate from TSV list
  ds=HTRLineDataset( from_tsv_file='dataset/htr_line_ds/train_ds.tsv' )
  ```

+ as a full directory:

  ```python
  ds=HTRLineDataset( from_work_folder='dataset/htr_line_ds/train')
  ```


<!--![](doc/_static/8257576.png)-->


### 2. Training from compiled line samples 

The training script assumes that there already exists a directory (default: `./data/current_working_set`) that contains all line images and transcriptions, as obtained by the step described above. There are several ways to initialize the training session:

#### 2.1 A list of training/validation line images

```bash
PYTHONPATH=. ./bin/ddp_htr_train.py -img_paths ./dataset/htr_line_ds/*.png -to_tsv 1
```

The script takes care of splitting all relevant images and metadata files into training, validation, and test subsets.

+ the optional `-to-tsv` flag allows for those subsets to be serialized into the images parent directory.


#### 2.2 A directory of line images

```bash
PYTHONPATH=. ./bin/ddp_htr_train.py -dataset_path dataset/htr_line_ds
```

By default, the script takes care of splitting all relevant images and metadata files into training, validation, and test subsets.

+ the optional `-to-tsv` flag allows for those subsets to be serialized into the images parent directory.
+ alternatively, with the `-from_tsv` flag, the set splitting step is skipped and the training subsets are constructed from the TSV lists in the directory.



<!--The training script only uses `libs/data.py` module to load the sample from this location; the lists of samples to be included in the training, validation, and test subsets is stored in the corresponding TSV files (`charters_ds_train.tsv`, `charters_ds_validate.tsv`, and `charters_ds_test.tsv`, respectively). If the TSV list has an extra field for a flat channel file (`*.npy.gz`), it is automatically concatenated to the tensor at loading time.-->

<!--For a charter dataset to play with, look at the [Data section](#Data) above.-->


#### 2.3 Usual options

To learn about the usual parameters of a training sessions (epochs, patience, etc.) run:

```bash
python3 ./bin/ddp_htr_train.py -h
```

Example:

```bash	
python3 ./bin/ddp_htr_train.py -batch_size 8 -max_epoch 100 -validation_freq 1 -dataset_path dataset/htr_line_ds
```


### 3. Inference

```bash
python3 ./bin/ddpa_htr_inference.py [ -<option> ... ]
```

where optional flags are one or more of the following:


#### Example:

```bash
export PYTHONPATH=$HOME/graz/htr/vre/ddpa_htr ./bin/ddp_htr_inference.py -model_path /tmp/model_save.mlmodel -img_paths */*/*/*.img.jpg -segmentation_file_suffix 'lines.pred.json
```


### 4. Additional scripts and modules


The following scripts are one-offs or deprecated. They are not meant for public consumption:

+ `bin/ddp_htr_train_with_abbrev.py`: (for experiments) training script that uses abbreviation masks on the GT transcriptions, as well as a custom edit distance function, in order to evaluate the abbreviations contribution to the CER.
+ `bin/ddp_htr_viewer.py`: visualizing confidence scores for a given HTR (color + transcription overlay)
+ some of the local modules are not part of the core HTR dependencies. For example:
