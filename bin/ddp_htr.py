#!/usr/bin/env python3

"""
HTR inference on page, with segmentation provided.
"""

from pathlib import Path
import sys
import fargv
from kraken import rpred
from kraken.lib import xml, models
import re
import glob
from PIL import Image

root = str( Path(__file__).parents[1] ) 
sys.path.append( root ) 

import seglib

p = {

    "appname": "htr",
    "model_path": Path( root, 'models', 'htr', 'default.mlmodel' ),
    "img_paths": set(glob.glob( str(Path.home().joinpath("tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*.jpg")))),
    "segmentation_dir": "", # for testing purpose
    "segmentation_file_suffix": "lines.pred", # under each image dir, suffix of the subfolder that contains the segmentation data 
        }




if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    for path in list( args.img_paths):
        print(path)


        stem = re.sub(r'\..+', '',  Path( path ).name )

        # 1. Load segmentation object from PageXML (better: JSON?)
        segmentation_dir = Path( path ).parent

        if args.segmentation_dir != "":
            if Path( args.segmentation_dir ).exists():
                segmentation_dir = Path( args.segmentation_dir )
            else:
                print("Provided segmentation directory {args.segmentation_dir} does not exists!")

        segmentation_object = None

        # Look for existing segmentation file (pattern: <image file stem>.lines.pred.{xml,json})
        segmentation_file_path_candidates = [ segmentation_dir.joinpath(f'{stem}.{args.segmentation_file_suffix}.{format_suffix}') for format_suffix in ('xml',) ]

        for segmentation_file_path in segmentation_file_path_candidates:
            if not segmentation_file_path.exists():
                continue
            segmentation_object = xml.XMLPage( segmentation_file_path ).to_container()
            break
        if segmentation_object is None:
            print("Could not find a proper segmentation data file. Aborting.")
            sys.exit()
        #else:
        #    print("Create segmentation record {} from segmentation data {}".format( segmentation_object, segmentation_file_path ))
         

        # 2. HTR inference
        line_ids =  [ line.id for line in segmentation_object.lines ]

        htr_model = models.load_any( args.model_path )
        pred_it = rpred.rpred( 
                        network = htr_model, 
                        im = Image.open( path ), 
                        bounds = segmentation_object )

        for line, line_id, record in zip(range(1, len(line_ids)+1), line_ids, pred_it):
            print(f'{line}\t{line_id}\t{record}')


