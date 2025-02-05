# HTR app

Prototype of a Kraken/VGSL-based, HTR app. The current draft contains:

- a model builder/wrapper `model_htr.py`, that provides the interfaces needed for training and inference.
- a trainer script `ddp_htr_train.py`, that uses the (provisory) Monasterium handwriting dataset
- a Kraken-based, high-level script `ddp_htr_inference.py` that runs inference on FSDB images, each provided with an existing JSON segmentation file, whose default path is ``<input_image_stem>.lines.pred.json``.
- a shell-script wrapper, that manages the dependencies for the HTR task, if ever needed: it relies on a Makefile to generate the segmentation meta-data for the charter, if they are not already present (with `ddpa_lines`) and calls then `ddp_htr_inference.py`.

TODO: 

+ HTR output keeps line coordinates
+ online retrieval of charters images
+ decoding options



## How to use

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


### Training

#### Recommended: preprocess first, then train

The `libs/charters_htr.py` class provides generic functionalities for downloading existing charters datasets and compiling them into line samples, on-disk and in-memory. However, because this preprocessing step is costly, the 
current training script assumes that there already exists a directory (default: `./data/current_working_set`) that contains all line images and transcriptions, as well as 3 TSV files, one of each of the training, validation, and test subsets.
The training script only uses `libs/charters_htr.py` module to load the sample from this location. If the directory contains an extra channel for a given image (*.npy.gz) -also listed in the TSV-it is automatically concatenated to the tensor at loading time.

A sample TSV:

```tsv
Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l1.png     Üllein Müllner von Gemünd       133     770     Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l1.channel.npy.gz
Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l2.png     Els Witbelein sein Swester      132     733     Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l2.channel.npy.gz
Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l3.png     der Öheim von Gemünd    134     681     Rst_Nbg-Briefbücher-Nr_2_0003_left-r1l3.channel.npy.gz
...
```

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

### Additional scripts

The following scripts are one-offs, that are not meant for public consumption:

+ `bin/ddp_htr_train_with_abbrev.py`: (for experiments) training script that uses abbreviation masks on the GT transcriptions, as well as a custom edit distance function, in order to evaluate the abbreviations contribution to the CER.
+ `bin/ddp_htr_viewer.py`: visualizing confidence scores for a given HTR (color + transcription overlay)
