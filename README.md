# HTR app

Embryo of a Kraken-based, HTR-only app. The current draft contains:

- a model builder/wrapper `model_htr.py`, that provides the interfaces needed for training and inference.
- a trainer script `ddp_htr_train.py`, that uses the (provisory) Monasterium handwriting dataset
- a Kraken-based, high-level script `ddp_htr_inference.py` that runs inference on page images, each provided with an existing (PageXML) segmentation file, whose default path is ``<input_image_stem>.lines.pred.xml``.
- a shell-script wrapper, that manage the dependencies for the HTR task: it generates the segmentation meta-dat for the charter, if they are not already present (with `ddpa_lines`) and calls then `ddp_htr_inference.py`.

TODO: 

+ decoding options

Examples:
	
```python	
python3 ./bin/ddp_htr_train.py -batch_size=4 -img_height=128 -img_width=3200 -max_epoch=10 -learning_rate=1e-2
```


## Syntax

~~~~.bash
-appname=<class 'str'>  Default 'htr_train' . Passed 'htr_train'
-batch_size=<class 'int'>  Default 2 . Passed 2
-img_height=<class 'int'>  Default 128 . Passed 128
-img_width=<class 'int'>  Default 2048 . Passed 2048
-max_epoch=<class 'int'>  Default 200 . Passed 200
-dataset_path_train=<class 'pathlib.PosixPath'> TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv). Default PosixPath('tests/data/polygons/monasterium_ds_train.tsv') . Passed PosixPath('tests/data/polygons/monasterium_ds_train.tsv')
-dataset_path_validate=<class 'pathlib.PosixPath'> TSV file containing the image paths and transcriptions. The parent folder is assumed to contain both the images and the alphabet (alphabet.tsv). Default PosixPath('tests/data/polygons/monasterium_ds_validate.tsv') . Passed PosixPath('tests/data/polygons/monasterium_ds_validate.tsv')
-learning_rate=<class 'float'>  Default 0.001 . Passed 0.001
-dry_run=<class 'bool'> Iterate over the batches once, but do not run the network. Default False . Passed False
-validation_freq=<class 'int'>  Default 100 . Passed 100
-save_freq=<class 'int'>  Default 100 . Passed 100
-resume_fname=<class 'str'>  Default 'model_save.ml' . Passed 'model_save.ml'
-mode=<class 'str'>  Default 'train' . Passed 'train'
-help=<class 'bool'> Print help and exit. Default False . Passed False
-bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False . Passed False
-h=<class 'bool'> Print help and exit Default False . Passed True
-v=<class 'int'> Set verbosity level. Default 1 . Passed 1
~~~~~~~~~~

