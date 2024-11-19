# HTR app

Prototype of a Kraken/VGSL-based, HTR app. The current draft contains:

- a model builder/wrapper `model_htr.py`, that provides the interfaces needed for training and inference.
- a trainer script `ddp_htr_train.py`, that uses the (provisory) Monasterium handwriting dataset
- a Kraken-based, high-level script `ddp_htr_inference.py` that runs inference on page images, each provided with an existing JSON segmentation file, whose default path is ``<input_image_stem>.lines.pred.json``.
- a shell-script wrapper, that manage the dependencies for the HTR task: it relies on a Makefile to generate the segmentation meta-data for the charter, if they are not already present (with `ddpa_lines`) and calls then `ddp_htr_inference.py`.

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
-appname=<class 'str'>  Default 'ddpa_htr' . 
-model_path=<class 'str'>  Default '/tmp/model_monasterium-2024-10-28.mlmodel' .
-img_paths=<class 'set'>  Default set() .
-charter_dirs=<class 'set'>  Default {'./'} .
-segmentation_dir=<class 'str'> Alternate location to search for the image segmentation data files (for testing). Default '' .
-segmentation_file_suffix=<class 'str'>  Default 'lines.pred.json' .
-output_dir=<class 'str'> Where the predicted transcription (a JSON file) is to be written. Default: in the parent folder of the charter image. Default '' .
-htr_file_suffix=<class 'str'>  Default 'htr.pred' .
-output_format=<class 'tuple'> Output format: 'stdout' for sending decoded lines on the standard output; 'json' and 'tsv' create JSON and TSV files, respectively. Default ('json', 'stdout', 'tsv') .
-padding_style=<class 'tuple'> How to pad the bounding box around the polygons: 'median'= polygon's median value, 'noise'=random noise, 'zero'=0-padding, 'none'=no padding Default ('median', 'noise', 'zero', 'none') .
-help=<class 'bool'> Print help and exit. Default False .
-bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False .
-h=<class 'bool'> Print help and exit Default False .
-v=<class 'int'> Set verbosity level. Default 1 .
```

### Training


```bash	
python3 ./bin/ddp_htr_train.py [ -<option> ...]
```

where optional flags are one or more of the following:

```bash
-appname=<class 'str'>  Default 'htr_train' .
-batch_size=<class 'int'>  Default 2 .
-img_height=<class 'int'>  Default 128 .
-img_width=<class 'int'>  Default 2048 .
-max_epoch=<class 'int'>  Default 200 .
-dataset_path_train=<class 'str'> TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv). 
-dataset_path_validate=<class 'str'> TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv). 
-learning_rate=<class 'float'>  Default 0.001 .
-dry_run=<class 'bool'> Iterate over the batches once, but do not run the network. Default False .
-validation_freq=<class 'int'>  Default 100 .
-save_freq=<class 'int'>  Default 100 .
-resume_fname=<class 'str'>  Default 'model_save.mlmodel' .
-mode=<class 'tuple'>  Default ('train', 'validate', 'test') .
-auxhead=<class 'bool'> Combine output with CTC shortcut Default False .
-help=<class 'bool'> Print help and exit. Default False .
-bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False .
-h=<class 'bool'> Print help and exit Default False .
-v=<class 'int'> Set verbosity level. Default 1 .
```

Example:

```bash	
python3 ./bin/ddp_htr_train.py -batch_size=4 -img_height=128 -img_width=3200 -max_epoch=10 -learning_rate=1e-2
```
