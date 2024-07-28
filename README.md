# HTR app

HTR-only app: use an existing segmentation, whose default path is ``<input_image_stem>.lines.pred.xml``

Examples::
	
	
	p3 bin/ddp_htr.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -model_path $HOME/tmp/models/htr/Tridis_by_Torres-Aguilar_2024/Tridis_Medieval_EarlyModern.mlmodel


## Syntax

+ appname=<class 'str'>  Default 'htr' . Passed 'htr'
+ model_path=<class 'pathlib.PosixPath'>  Default PosixPath('/home/nicolas/tmp/models/htr/Tridis_by_Torres-Aguilar_2024/Tridis_Medieval_EarlyModern.mlmodel') . Passed PosixPath('/home/nicolas/tmp/models/htr/Tridis_by_Torres-Aguilar_2024/Tridis_Medieval_EarlyModern.mlmodel')
+ img_paths=<class 'set'>  Default {'/home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/147c32f12ef7b285bd19e44ab47e253a.img.jpg', '/home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/22e65eaa05f1cf1a71b5ccf29cfc0947.img.jpg'} . Passed {'/home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/147c32f12ef7b285bd19e44ab47e253a.img.jpg', '/home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/22e65eaa05f1cf1a71b5ccf29cfc0947.img.jpg'}
+ segmentation_dir=<class 'str'>  Default '' . Passed ''
+ segmentation_file_suffix=<class 'str'>  Default 'lines.pred' . Passed 'lines.pred'
+ help=<class 'bool'> Print help and exit. Default False . Passed False
+ bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False . Passed False
+ h=<class 'bool'> Print help and exit Default False . Passed True
+ v=<class 'int'> Set verbosity level. Default 1 . Passed 1

