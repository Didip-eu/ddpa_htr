# HTR app

Embryo of a Kraken-based, HTR-only app: the current draft runs inference on a set of chart images, each provided with an existing (PageXML) segmentation file, whose default path is ``<input_image_stem>.lines.pred.xml``.

TODO: 

+ better output
+ passing set of line images
+ decoding options
+ performance measure when GT file exists

Examples:
	
```python	

# Explicit recognition model
python3 bin/ddp_htr.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -model_path $HOME/tmp/models/htr/Tridis_by_Torres-Aguilar_2024/Tridis_Medieval_EarlyModern.mlmodel

# Using the default model (Tridis)
python3 bin/ddp_htr.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg 
```


## Syntax

+ appname=<class 'str'>  Default 'htr' . Passed 'htr'
+ model_path=<class 'pathlib.PosixPath'>  Default PosixPath('/home/nicolas/tmp/models/htr/Tridis_by_Torres-Aguilar_2024/Tridis_Medieval_EarlyModern.mlmodel') . 
+ img_paths=<class 'set'>  Default {'/home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*.img.jpg' }
+ segmentation_dir=<class 'str'>  Default '' . 
+ segmentation_file_suffix=<class 'str'>  Default 'lines.pred' . 
+ help=<class 'bool'> Print help and exit. Default False . 
+ h=<class 'bool'> Print help and exit Default False . Passed True
+ v=<class 'int'> Set verbosity level. Default 1 . Passed 1

