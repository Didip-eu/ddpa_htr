# HTR app

Prototype of a Kraken/VGSL-based, HTR app. The current draft contains:

- a model builder/wrapper `model_htr.py`, that provides the interfaces needed for training and inference.
- a training script `ddp_htr_train.py`, that can use any pre-processed set of line image+transcription samples, preferably stored on-disk (see below).
- a high-level script `ddp_htr_inference.py` that runs inference on FSDB images, each provided with an existing JSON segmentation file, whose default path is ``<input_image_stem>.lines.pred.json``.
- a shell-script wrapper, that manages the dependencies for the HTR task, if ever needed: it relies on a Makefile to generate the segmentation meta-data for the charter, if they are not already present (with `ddpa_lines`) and calls then `ddp_htr_inference.py`.

TODO: 

+ HTR output keeps line coordinates
+ online retrieval of charters images
+ decoding options
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


### Training

#### Precompiling the dataset pages

The `libs/charters_htr.py` module provides generic functionalities for handling charters datasets, including downloading/compiling a few specific datasets into line samples, on-disk and in-memory. 
For instance, the following Python call compiles line samples from an existing directory containing page images and their meta-data:


```python
sys.path.append('.')
from libs import charters_htr, maskutils as msk
charters_htr.ChartersDataset(from_page_dir=f'{os.environ["HOME"]}/tmp/data/Monasterium/MonasteriumTekliaGTDataset', work_folder='./data/MonasteriumTeklia', channel_func=msk.bbox_blurry_channel)
```

The resulting work folder (it is created if it does not exist) stores 

+ the line images  (`*.png`)
+ the transcription ground truth files (`*.gt.txt`)
+ binary masks generated from each line polygon boundaries (`*.bool.npy.gz`), to be used at loading time with a masking function of choice
+ in this particular case, for each line image,  an extra flat, gray channel that is to be concatenated to the image tensor at loading time
+ a TSV file `charters_ds_train.tsv` that lists all samples in the training subset, with 70% of the total number of samples, as shown below:

```tsv
Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l1.png     Üllein Müllner von Gemünd       133     770     Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l1.channel.npy.gz
Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l2.png     Els Witbelein sein Swester      132     733     Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l2.channel.npy.gz
Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l3.png     der Öheim von Gemünd    134     681     Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l3.channel.npy.gz
...
```

At this point, all sample data have been generated. To obtain the validation and test sets, we simply read from the existing work folder. The  command generates 3 TSV files for the training, validation, and test subsets, respectively:

```python
myTrainingSet=charters_htr.ChartersDataset(from_work_folder='/home/nicolas/ddpa_htr/data/MonasteriumTeklia', line_padding_style='noise', subset_ratios={.8, .1, .1})
```
The `line_padding_style` keyword specifies how the polygon mask should be used at loading time (other options: `median`, `zero`, or `none`).
By default, the constructor above returns a training set object, but any particular live subset can be loaded:
```python
myTestSet=charters_htr.ChartersDataset(from_work_folder='/home/nicolas/ddpa_htr/data/MonasteriumTeklia', line_padding_style='noise', subset='test')
```

For further options in  `libs/charters_htr.py` module, look at the embedded documentation or at the mother repo [DiDip_handwriting_datasets](https://github.com/Didip-eu/didip_handwriting_datasets).


#### Training from compiled line samples

Because the preprocessing step is costly and in order to avoid unwanted complexity into the training logic, the 
current training script assumes that there already exists a directory (default: `./data/current_working_set`) that contains all line images and transcriptions, as well as 3 TSV files, one for each of the training, validation, and test subsets.
The training script only uses `libs/charters_htr.py` module to load the sample from this location; the lists of samples to be included in the training, validation, and test subsets is stored in the corresponding TSV files (`charters_ds_train.ds`, `charters_ds_validate.tsv`, and `charters_ds_test.tsv`, respectively). If the TSV list has an extra field for a flat channel file (`*.npy.gz`), it is automatically concatenated to the tensor at loading time.

For a charter dataset to play with, look at the [Data section](#Data) above.


#### Syntax

```bash	
python3 ./bin/ddp_htr_train.py [ -<option> ...]
```

where optional flags are one or more of the following:

```bash
-appname=<class 'str'>  Default 'htr_train' . Passed 'htr_train'
-batch_size=<class 'int'>  Default 2 . Passed 2
-img_height=<class 'int'>  Default 128 . Passed 128
-img_width=<class 'int'>  Default 2048 . Passed 2048
-max_epoch=<class 'int'>  Default 200 . Passed 200
-dataset_path_train=<class 'str'> TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions. Default 'data/current_working_set/charters_ds_train.tsv' . Passed 'data/current_working_set/charters_ds_train.tsv'
-dataset_path_validate=<class 'str'> TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions. Default 'data/current_working_set/charters_ds_validate.tsv' . Passed 'data/current_working_set/charters_ds_validate.tsv'
-dataset_path_test=<class 'str'> TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both images and transcriptions. Default 'data/current_working_set/charters_ds_test.tsv' . Passed 'data/current_working_set/charters_ds_test.tsv'
-ignored_chars=<class 'list'>  Default [] . Passed []
-learning_rate=<class 'float'>  Default 0.001 . Passed 0.001
-dry_run=<class 'bool'> Iterate over the batches once, but do not run the network. Default False . Passed False
-validation_freq=<class 'int'>  Default 1 . Passed 1
-save_freq=<class 'int'>  Default 1 . Passed 1
-resume_fname=<class 'str'> Model *.mlmodel to load. By default, the epoch count will start from the epoch that has been last stored in this file's meta-data. To ignore this and reset the epoch count, set the -reset_epoch option. Default 'model_save.mlmodel' . Passed 'model_save.mlmodel'
-reset_epochs=<class 'bool'> Ignore the the epoch data stored in the model file - use for fine-tuning an existing model on a different dataset. Default False . Passed False
-mode=<class 'tuple'>  Default ('train', 'validate', 'test') . Passed 'train'
-auxhead=<class 'bool'> ([BROKEN]Combine output with CTC shortcut Default False . Passed False
-help=<class 'bool'> Print help and exit. Default False . Passed False
-bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False . Passed False
-h=<class 'bool'> Print help and exit Default False . Passed True
-v=<class 'int'> Set verbosity level. Default 1 . Passed 1
```

#### Example:

```bash	
python3 ./bin/ddp_htr_train.py -batch_size 8 -max_epoch 100 -validation_freq 1
```
### Inference

```bash
python3 ./bin/ddpa_htr_inference.py [ -<option> ... ]
```

where optional flags are one or more of the following:

```bash
-appname=<class 'str'>  Default 'ddpa_htr_inference' . Passed 'ddpa_htr_inference'
-model_path=<class 'str'>  Default '/tmp/model_monasterium-2024-10-28.mlmodel' . Passed '/tmp/model_monasterium-2024-10-28.mlmodel'
-img_paths=<class 'set'>  Default set() . Passed set()
-charter_dirs=<class 'set'>  Default set() . Passed set()
-segmentation_dir=<class 'str'> Alternate location to search for the image segmentation data files (for testing). Default '' . Passed ''
-segmentation_file_suffix=<class 'str'>  Default 'lines.pred.json' . Passed 'lines.pred.json'
-output_dir=<class 'str'> Where the predicted transcription (a JSON file) is to be written. Default: in the parent folder of the charter image. Default '' . Passed ''
-htr_file_suffix=<class 'str'>  Default 'htr.pred' . Passed 'htr.pred'
-output_format=<class 'tuple'> Output format: 'stdout' for sending decoded lines on the standard output; 'json' and 'tsv' create JSON and TSV files, respectively. Default ('json', 'stdout', 'tsv') . Passed 'json'
-output_data=<class 'tuple'> By default, the application yields character predictions; 'logits' have it returns logits instead. Default ('pred', 'logits', 'all') . Passed 'pred'
-padding_style=<class 'tuple'> How to pad the bounding box around the polygons: 'median'= polygon's median value, 'noise'=random noise, 'zero'=0-padding, 'none'=no padding Default ('median', 'noise', 'zero', 'none') . Passed 'median'
-help=<class 'bool'> Print help and exit. Default False . Passed False
-bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False . Passed False
-h=<class 'bool'> Print help and exit Default False . Passed True
-v=<class 'int'> Set verbosity level. Default 1 . Passed 1
```

#### Example:

```bash
export PYTHONPATH=$HOME/graz/htr/vre/ddpa_htr ./bin/ddp_htr_inference.py -model_path /tmp/model_save.mlmodel -img_paths */*/*/*.img.jpg -segmentation_file_suffix 'lines.pred.json
```


### Additional scripts and modules


The following scripts are one-offs, that are not meant for public consumption:

+ `bin/ddp_htr_train_with_abbrev.py`: (for experiments) training script that uses abbreviation masks on the GT transcriptions, as well as a custom edit distance function, in order to evaluate the abbreviations contribution to the CER.
+ `bin/ddp_htr_viewer.py`: visualizing confidence scores for a given HTR (color + transcription overlay)
+ some of the local modules are not part of the core HTR dependencies. For example:
  * pre-processing/compilation step: `seglib.py`, `download__utils.py`
  * for convenience only: `maskutils.py`, `visuals.py`
